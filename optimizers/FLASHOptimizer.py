import numpy as np
from sklearn.tree import DecisionTreeRegressor

from optimizers.base_optimizer import BaseOptimizer
from models.Data import Data
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    OrdinalHyperparameter,
    Constant,
)


class FLASHOptimizer(BaseOptimizer):
    """
    FLASH (single-objective acquisition; NO Bazza) in your framework:

    - Uses ConfigSpace sampling for random init and candidate pools
    - Projects to nearest dataset row via KD-tree (Data)
    - Evaluates via model_wrapper.run_model() (your framework handles multiobjective scalarization)
    - Trains CART surrogate on evaluated points
    - Picks next config as argmin predicted fitness (or argmax score equivalently)
    - Caches projected evaluations
    - Logs via logging_util.log(valid_hp, fitness, 1)
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)

        self.nn = Data(
            self.X_df.values.tolist(),
            column_types=self.model_config.column_types,
        )

        self.config_space, self.param_names, _ = self.model_config.get_configspace()

        self.cache = {}
        self.best_config = None
        self.best_value = None

    # --------------------- Helpers --------------------- #

    def _clean(self, v):
        return v.item() if hasattr(v, "item") else v

    def _nearest_row(self, hp_dict):
        query = [hp_dict[c] for c in self.columns]
        row = self.nn.nearestRow(query)
        return {c: self._clean(v) for c, v in zip(self.columns, row)}

    def _row_tuple(self, hp_dict):
        return tuple(hp_dict[c] for c in self.columns)

    def _sample_random_config(self):
        """
        Sample a *full* config dict keyed by self.columns.
        This must align with your encoded columns, since you later build vectors
        in that same column order.
        """
        sample = {}
        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, Constant):
                sample[hp.name] = hp.value
            elif isinstance(hp, OrdinalHyperparameter):
                sample[hp.name] = np.random.choice(list(hp.sequence))
            elif isinstance(hp, CategoricalHyperparameter):
                sample[hp.name] = np.random.choice(list(hp.choices))
            else:
                raise ValueError(f"Unsupported HP type {type(hp)}")

        # Ensure all required columns exist (some configs may have constants only)
        for c in self.columns:
            if c not in sample:
                raise ValueError(
                    f"FLASH sampled config missing column '{c}'. "
                    f"ConfigSpace must include hyperparameter names matching encoded columns."
                )
        return sample

    def _evaluate(self, hp_dict):
        """
        Measure(config) in your framework:
        - project to nearest real row
        - cache by projected row
        - call run_model(valid_hp)
        - convert to fitness (minimize)
        """
        valid_hp = self._nearest_row(hp_dict)
        key = self._row_tuple(valid_hp)

        if key in self.cache:
            fitness = self.cache[key]
        else:
            score = self.model_wrapper.run_model(valid_hp)
            fitness = 1 - score
            self.cache[key] = fitness

        self.logging_util.log(valid_hp, fitness, 1)
        return fitness, valid_hp

    # --------------------- Optimize --------------------- #

    def optimize(self):
        if not self.logging_util:
            raise ValueError("Logging util not provided.")

        # Budget = total number of measurements
        budget = int(self.config["n_trials"])
        init_samples = int(self.config.get("init_samples", 4))
        candidate_pool = int(self.config.get("candidate_pool", 1000))

        if init_samples <= 0 or init_samples > budget:
            raise ValueError("init_samples must be in [1, n_trials].")

        rng = np.random.default_rng(self.seed)

        self.logging_util.start_logging()

        surrogate = DecisionTreeRegressor(random_state=self.seed)

        # ---------------- Initial random measurements ---------------- #
        X = []
        y = []
        best_f = float("inf")
        best_hp = None

        # Keep sampling until we collect init_samples UNIQUE projected rows
        seen = set()
        while len(X) < init_samples:
            cfg = self._sample_random_config()
            vec = [cfg[c] for c in self.columns]

            # Evaluate (with projection)
            fitness, valid_hp = self._evaluate(cfg)
            key = self._row_tuple(valid_hp)

            if key in seen:
                continue
            seen.add(key)

            X.append([valid_hp[c] for c in self.columns])  # train on projected row
            y.append(fitness)

            if fitness < best_f:
                best_f = fitness
                best_hp = valid_hp

        X = np.array(X, dtype=object)
        y = np.array(y, dtype=float)

        # ---------------- Main FLASH loop ---------------- #
        while len(y) < budget:
            # Train CART on evaluated points
            surrogate.fit(X, y)

            # Sample a candidate pool (membership-query style), then pick best predicted
            cand_cfgs = []
            cand_vecs = []
            cand_keys = []

            # Keep filling pool with UNIQUE projected rows not yet seen
            attempts = 0
            while len(cand_vecs) < candidate_pool and attempts < candidate_pool * 50:
                attempts += 1
                cfg = self._sample_random_config()

                # Project now, so surrogate predictions correspond to real rows
                valid_hp = self._nearest_row(cfg)
                key = self._row_tuple(valid_hp)
                if key in seen:
                    continue

                cand_cfgs.append(valid_hp)  # already projected dict
                cand_vecs.append([valid_hp[c] for c in self.columns])
                cand_keys.append(key)

            # If we can't find any new candidates, stop early
            if not cand_vecs:
                break

            cand_X = np.array(cand_vecs, dtype=object)
            preds = surrogate.predict(cand_X)

            # Acquisition (single-objective): pick best predicted (min fitness)
            best_idx = int(np.argmin(preds))
            next_hp = cand_cfgs[best_idx]
            next_key = cand_keys[best_idx]

            # Evaluate (next_hp is already projected, but _evaluate expects dict keyed by columns)
            # _evaluate will re-project; that's fine and stable.
            fitness, valid_hp = self._evaluate(next_hp)
            key = self._row_tuple(valid_hp)

            # Add to training data if new
            if key in seen:
                continue
            seen.add(key)

            X = np.vstack([X, np.array([[valid_hp[c] for c in self.columns]], dtype=object)])
            y = np.append(y, float(fitness))

            if fitness < best_f:
                best_f = fitness
                best_hp = valid_hp

        # ---------------- Final best ---------------- #
        final_score = 1 - best_f
        self.best_config = best_hp
        self.best_value = final_score

        self.logging_util.stop_logging()
        return self.best_config, self.best_value