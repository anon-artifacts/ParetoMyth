import random
import numpy as np
from optimizers.base_optimizer import BaseOptimizer
from models.Data import Data


class AroundOptimizer(BaseOptimizer):
    """
    FAST + FAIR + FAITHFUL AROUND optimizer (pure Python)

    - No numba
    - No full distance precompute
    - Lua-faithful kmeans++ behavior
    - Correct diversity preservation
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        self.best_config = None
        self.best_value = None

        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)

        self.rows = self.X_df.values.tolist()

        self.nn = Data(
            self.rows,
            column_types=self.model_config.column_types,
        )

        random.seed(seed)
        np.random.seed(seed)

    # -----------------------------
    # Helpers
    # -----------------------------
    def _clean(self, v):
        if isinstance(v, (np.integer, np.int64)):
            return int(v)
        if isinstance(v, (np.floating, np.float64)):
            return float(v)
        return v

    def _row_to_hp(self, row):
        return {c: self._clean(v) for c, v in zip(self.columns, row)}

    # -----------------------------
    # FAST + CORRECT AROUND
    # -----------------------------
    def _around(self, rows, budget, sample_size):
        n = len(rows)
        if n == 0:
            return []

        nn = self.nn

        # Distance cache (Lua-style)
        dist_cache = {}

        def cached_dist(i, j):
            if i > j:
                i, j = j, i
            key = (i, j)
            d = dist_cache.get(key)
            if d is None:
                d = nn.xdist(rows[i], rows[j])
                dist_cache[key] = d
            return d

        # chosen centers (indices)
        chosen = []
        chosen_set = set()

        first = random.randrange(n)
        chosen.append(first)
        chosen_set.add(first)

        # min distance to any chosen center
        min_d = [float("inf")] * n

        for _ in range(1, budget):
            last = chosen[-1]

            # update min distances ONCE (critical!)
            for i in range(n):
                d = cached_dist(i, last)
                if d < min_d[i]:
                    min_d[i] = d

            total = 0.0
            candidates = []

            for _ in range(min(sample_size, n)):
                i = random.randrange(n)
                d2 = min_d[i] * min_d[i]
                if d2 > 0:
                    candidates.append((i, d2))
                    total += d2

            if total <= 0:
                break

            # weighted random choice
            r = random.random() * total
            acc = 0.0
            pick = None

            for i, w in candidates:
                acc += w
                if acc >= r:
                    pick = i
                    break

            if pick is None:
                pick = candidates[-1][0]

            if pick not in chosen_set:
                chosen.append(pick)
                chosen_set.add(pick)

        return [rows[i] for i in chosen]

    # -----------------------------
    # OPTIMIZE
    # -----------------------------
    def optimize(self):
        budget = self.config["n_trials"]
        sample_size = min(32, len(self.rows))

        print(f"=== Running AROUND (pure Python, faithful), budget={budget} ===")

        selected_rows = self._around(
            rows=self.rows,
            budget=budget,
            sample_size=sample_size,
        )

        self.logging_util.start_logging()

        best_hp = None
        best_score = -float("inf")

        for row in selected_rows:
            hp = self._row_to_hp(row)
            score = self.model_wrapper.run_model(hp)
            fitness = 1 - score

            self.logging_util.log(hp, fitness, 1)

            if score > best_score:
                best_score = score
                best_hp = hp

        self.logging_util.stop_logging()

        self.best_config = best_hp
        self.best_value = 1-best_score

        print(f"Best config = {self.best_config}")
        print(f"Best score  = {self.best_value}")

        return self.best_config, self.best_value
