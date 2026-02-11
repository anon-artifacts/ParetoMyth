import numpy as np
import time
import uuid
import random
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    OrdinalHyperparameter,
    Constant
)
from optimizers.base_optimizer import BaseOptimizer
from models.Data import Data


class BOCAOptimizer(BaseOptimizer):
    """
    BOCA implementation under the same clean modern framework as SMACOptimizer.

    ✔ Works over discrete ConfigSpace
    ✔ Converts configs → nearest table row
    ✔ Evaluates using model_wrapper
    ✔ Surrogate model = RandomForest
    ✔ EI acquisition
    ✔ BOCA structured search:
        - Top-K impactful params
        - 2^K exhaustive important settings
        - C random settings of unimportant params
        - Combine important × unimportant
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)

        # KD tree for nearest valid configuration
        self.nn = Data(
            self.X_df.values.tolist(),
            column_types=self.model_config.column_types
        )

        self.config_space, self.param_names, _ = self.model_config.get_configspace()

        self.cache = {}

        # BOCA parameters
        self.init_samples = self.config.get("init_samples", 20)
        self.K = self.config.get("top_k", 8)                     # BOCA top-K impactful opts
        self.decay = self.config.get("decay", 0.5)
        self.scale = self.config.get("scale", 10)
        self.offset = self.config.get("offset", 20)

        self.best_config = None
        self.best_value = None

    # ----------------------------------------
    # Utility helpers
    # ----------------------------------------

    def _clean(self, v):
        if hasattr(v, "item"): return v.item()
        return v

    def _nearest_row(self, hp_dict):
        q = [hp_dict[c] for c in self.columns]
        r = self.nn.nearestRow(q)
        return {c: self._clean(v) for c, v in zip(self.columns, r)}

    def _row_tuple(self, hp_dict):
        return tuple(hp_dict[c] for c in self.columns)

    def _sample_random_cfg(self):
        d = {}
        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, Constant):
                d[hp.name] = hp.value
            elif isinstance(hp, OrdinalHyperparameter):
                d[hp.name] = random.choice(list(hp.sequence))
            elif isinstance(hp, CategoricalHyperparameter):
                d[hp.name] = random.choice(list(hp.choices))
            else:
                raise ValueError(f"Unsupported HP type: {type(hp)}")
        return d

    def _evaluate(self, hp_dict):
        valid = self._nearest_row(hp_dict)
        key = self._row_tuple(valid)

        if key in self.cache:
            return self.cache[key]

        score = self.model_wrapper.run_model(valid)
        fitness = 1 - score
        self.cache[key] = fitness

        if self.logging_util:
            self.logging_util.log(valid, fitness, 1)

        return fitness

    # ----------------------------------------
    # EI acquisition
    # ----------------------------------------
    def _ei(self, preds, eta):
        preds = np.array(preds).T
        mean = preds.mean(axis=1)
        std = preds.std(axis=1)
        std_safe = np.where(std == 0, 1e-8, std)

        z = (eta - mean) / std_safe
        ei = (eta - mean) * norm.cdf(z) + std_safe * norm.pdf(z)
        ei[std == 0] = 0

        return ei

    # ------------------------------------------------------
    # BOCA candidate generation
    # ------------------------------------------------------
    def _enumerate_important_settings(self, important):
        """Return all 2^K bit assignments for important optimizations."""
        K = len(important)
        settings = []

        for i in range(2 ** K):
            bits = format(i, f"0{K}b")
            cfg = {important[j]: int(bits[j]) for j in range(K)}
            settings.append(cfg)

        return settings

    def _sample_unimportant(self, unimportant, C):
        """Sample C random configurations of unimportant parameters."""
        samples = []
        for _ in range(C):
            cfg = {idx: random.choice([0, 1]) for idx in unimportant}
            samples.append(cfg)
        return samples

    # ------------------------------------------------------
    # Main BOCA optimizer loop
    # ------------------------------------------------------
    def optimize(self):

        if not self.logging_util:
            raise ValueError("logging_util required")

        self.logging_util.start_logging()
        n_trials = self.config["n_trials"]

        # ------------------------------------------------------
        # 1. Initial random samples
        # ------------------------------------------------------
        X_cfgs = []
        y_vals = []

        for _ in range(min(self.init_samples, n_trials)):
            cfg = self._sample_random_cfg()
            val = self._evaluate(cfg)
            X_cfgs.append(cfg)
            y_vals.append(val)

        best_idx = int(np.argmin(y_vals))
        best_cfg = X_cfgs[best_idx]
        best_val = y_vals[best_idx]

        # Precompute decay parameters
        sigma = -self.scale ** 2 / (2 * np.log(self.decay))

        # ------------------------------------------------------
        # 2. BOCA iterations
        # ------------------------------------------------------
        for t in range(len(X_cfgs), n_trials):

            # shrinking neighborhood parameter
            rnum = np.exp(
                -max(0, len(X_cfgs) - self.offset) ** 2 / (2 * sigma ** 2)
            )

            # Train surrogate
            X_vec = np.array([[cfg[p] for p in self.columns] for cfg in X_cfgs])
            y_arr = np.array(y_vals)
            rf = RandomForestRegressor()
            rf.fit(X_vec, y_arr)

            # Compute feature importance (impactful optimizations)
            importances = list(enumerate(rf.feature_importances_))
            importances = sorted(importances, key=lambda x: x[1], reverse=True)

            important = [idx for idx, _ in importances[:self.K]]
            unimportant = [idx for idx, _ in importances[self.K:]]

            # Enumerate all 2^K important settings
            important_settings = self._enumerate_important_settings(important)

            # Number of unimportant samples C
            C = max(1, int(rnum * len(unimportant)))

            # Build candidate list
            candidates = []
            for imp_cfg in important_settings:
                unimp_samples = self._sample_unimportant(unimportant, C)

                for u_cfg in unimp_samples:
                    full = {}

                    # Fill fixed important settings
                    for hp_idx, val in imp_cfg.items():
                        full[self.columns[hp_idx]] = val

                    # Fill sampled unimportant settings
                    for hp_idx, val in u_cfg.items():
                        full[self.columns[hp_idx]] = val

                    candidates.append(full)

            # Filter candidates that were already evaluated
            train_set = {self._row_tuple(self._nearest_row(c)) for c in X_cfgs}

            # Predict means/std from RF ensemble
            cand_vec = np.array([
                [cfg[p] for p in self.columns] for cfg in candidates
            ])

            preds = [tree.predict(cand_vec) for tree in rf.estimators_]
            ei_vals = self._ei(preds, best_val)

            # Rank candidates by EI
            ranked = sorted(
                zip(candidates, ei_vals),
                key=lambda x: x[1],
                reverse=True
            )

            # Pick next non-duplicate BOCA candidate
            for cfg, _ in ranked:
                if self._row_tuple(self._nearest_row(cfg)) not in train_set:
                    next_cfg = cfg
                    break

            # Evaluate candidate
            val = self._evaluate(next_cfg)
            X_cfgs.append(next_cfg)
            y_vals.append(val)

            if val < best_val:
                best_val = val
                best_cfg = next_cfg

        # Final nearest valid config
        final_cfg = self._nearest_row(best_cfg)
        final_score = 1 - self.model_wrapper.run_model(final_cfg)

        self.best_config = final_cfg
        self.best_value = final_score

        self.logging_util.stop_logging()
        return self.best_config, self.best_value
