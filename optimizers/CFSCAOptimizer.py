import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm

from optimizers.base_optimizer import BaseOptimizer
from models.Data import Data
from utils import DistanceUtil


class CFSCAOptimizer(BaseOptimizer):
    """
    Tabular-safe CFSCA:
    - Proposes configs
    - Projects to nearest dataset row
    - Evaluates ONLY valid rows
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        # ---- dataset ----
        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)

        self.nn = Data(
            self.X_df.values.tolist(),
            column_types=self.model_config.column_types,
        )

        # ---- objectives ----
        sample = self.X_df.iloc[0].to_dict()
        self.num_objectives = len(self.model_wrapper.get_score(sample))

        # ---- CFSCA params ----
        self.dim = len(self.columns)
        self.budget = self.config["n_trials"]

        # If related_idx exists, use it; otherwise fall back to all
        self.related_idx = getattr(model_config, "related_idx", list(range(self.dim)))

        self.eval_count = 0
        self.cache = {}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _nearest_row(self, hp_dict):
        query = [hp_dict[c] for c in self.columns]
        row = self.nn.nearestRow(query)
        return {c: v for c, v in zip(self.columns, row)}

    def _row_key(self, hp_dict):
        return tuple(hp_dict[c] for c in self.columns)

    def _score(self, hp_dict):
        key = self._row_key(hp_dict)
        if key in self.cache:
            return self.cache[key]

        scores = self.model_wrapper.get_score(hp_dict)
        self.cache[key] = scores
        return scores

    def _d2h(self, scores):
        ideal = [1] * len(scores)
        return DistanceUtil.d2h(ideal, scores)

    # ------------------------------------------------------------------
    # Expected Improvement
    # ------------------------------------------------------------------

    def _ei(self, preds, eta):
        preds = np.array(preds).transpose(1, 0)
        mu = np.mean(preds, axis=1)
        sigma = np.std(preds, axis=1)

        sigma[sigma == 0] = 1e-9
        z = (eta - mu) / sigma
        return (eta - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # ------------------------------------------------------------------
    # Initial sampling
    # ------------------------------------------------------------------

    def _random_row(self):
        idx = random.randint(0, len(self.X_df) - 1)
        return self.X_df.iloc[idx].to_dict()

    # ------------------------------------------------------------------
    # Main optimization loop
    # ------------------------------------------------------------------

    def optimize(self):
        X_train = []
        y_train = []

        # ---- initial seed ----
        while len(X_train) < 5 and self.eval_count < self.budget:
            hp = self._random_row()
            scores = self._score(hp)
            X_train.append([hp[c] for c in self.columns])
            y_train.append(self._d2h(scores))
            self.eval_count += 1

        # ---- main loop ----
        while self.eval_count < self.budget:
            model = RandomForestRegressor(random_state=self.seed)
            model.fit(np.array(X_train), np.array(y_train))

            # candidate pool (sampled rows)
            candidates = [
                self._random_row()
                for _ in range(min(2000, len(self.X_df)))
            ]

            X_cand = [[c[cx] for cx in self.columns] for c in candidates]
            preds = [t.predict(X_cand) for t in model.estimators_]

            eta = min(y_train)
            ei_vals = self._ei(preds, eta)

            best_idx = int(np.argmax(ei_vals))
            best_hp = candidates[best_idx]

            scores = self._score(best_hp)
            d2h = self._d2h(scores)

            X_train.append([best_hp[c] for c in self.columns])
            y_train.append(d2h)
            self.eval_count += 1

        # ---- final selection ----
        best_i = int(np.argmin(y_train))
        self.best_config = {
            c: X_train[best_i][i]
            for i, c in enumerate(self.columns)
        }
        self.best_value = y_train[best_i]

        return self.best_config, self.best_value
