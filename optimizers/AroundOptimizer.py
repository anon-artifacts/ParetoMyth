import numpy as np
import random
from math import sqrt
from sklearn.model_selection import train_test_split
from optimizers.base_optimizer import BaseOptimizer
from models.Data import Data


class AroundOptimizer(BaseOptimizer):
    """
    Pure AROUND optimizer:
    - Uses encoded training rows
    - Implements around() internally
    - No BL dependency
    - ActLearn-style evaluation on FULL TEST SET
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        self.best_config = None
        self.best_value = None

        # Encoded dataframe
        self.X_df = self.model_wrapper.X_encoded
        self.columns = list(self.X_df.columns)

        # --- Train/Test split ---
        self.X_train, self.X_test = train_test_split(
            self.X_df,
            test_size=0.5,
            random_state=self.seed,
            shuffle=True
        )

        # KD-trees for nearest-row correction
        self.nn_train = Data(
            self.X_train.values.tolist(),
            column_types=self.model_config.column_types
        )
        self.nn_test = Data(
            self.X_test.values.tolist(),
            column_types=self.model_config.column_types
        )

    # -------------------------------------------------------
    # Helper: Clean numpy types
    # -------------------------------------------------------
    def _clean(self, v):
        if isinstance(v, (np.integer, np.int64)): return int(v)
        if isinstance(v, (np.floating, np.float64)): return float(v)
        return v

    def _row_dict(self, row_list):
        return {col: self._clean(v) for col, v in zip(self.columns, row_list)}

    # -------------------------------------------------------
    # Convert row (list) → numeric vector
    # -------------------------------------------------------
    def _vec(self, row):
        return np.array(row, dtype=float)

    # -------------------------------------------------------
    # INTERNAL AROUND IMPLEMENTATION (Diversity sampler)
    # -------------------------------------------------------
    def _around(self, rows, budget, sample_size=32):
        """
        rows: list of encoded rows (lists)
        returns: list of *row-vectors* (not dicts)
        """

        rows = list(rows)
        n = len(rows)
        if n == 0:
            return []

        # Start with ONE random point
        chosen = [random.choice(rows)]
        chosen_vecs = [self._vec(chosen[0])]

        remaining = list(rows)

        # Remove the chosen one
        remaining.remove(chosen[0])

        for _ in range(1, budget):
            if not remaining:
                break

            # Sample a subset for efficiency
            k = min(sample_size, len(remaining))
            candidates = random.sample(remaining, k)

            best_candidate = None
            best_dist = -1

            for c in candidates:
                c_vec = self._vec(c)

                # compute distance to nearest chosen point
                d = np.min([np.linalg.norm(c_vec - z) for z in chosen_vecs])

                if d > best_dist:
                    best_dist = d
                    best_candidate = c

            chosen.append(best_candidate)
            chosen_vecs.append(self._vec(best_candidate))
            remaining.remove(best_candidate)

        return chosen

    # -------------------------------------------------------
    # Build ActLearn-style sorter model
    # -------------------------------------------------------
    def _build_sorter(self, selected_rows):
        """
        selected_rows: list of row dicts
        """

        if len(selected_rows) == 0:
            return lambda hp: 0

        b = int(sqrt(len(selected_rows)))
        b = max(1, min(b, len(selected_rows)))

        best = selected_rows[:b]
        rest = selected_rows[b:] if len(selected_rows) > b else best[:]

        def to_vec(hp):
            return np.array([hp[c] for c in self.columns], dtype=float)

        best_center = np.mean([to_vec(r) for r in best], axis=0)
        rest_center = np.mean([to_vec(r) for r in rest], axis=0)

        def sorter(hp_dict):
            v = to_vec(hp_dict)
            d_best = np.linalg.norm(v - best_center)
            d_rest = np.linalg.norm(v - rest_center)
            return d_best - d_rest

        return sorter

    # -------------------------------------------------------
    # OPTIMIZE
    # -------------------------------------------------------
    def optimize(self):
        budget = self.config["n_trials"]
        print(f"=== Running AROUND optimizer, budget={budget} ===")

        train_rows = self.X_train.values.tolist()

        # ----------------------------------------------
        # 1. Select TRAIN rows fully via internal around()
        # ----------------------------------------------
        selected_rows = self._around(train_rows, budget, sample_size=64)

        # ----------------------------------------------
        # 2. Evaluate selected TRAIN rows
        # ----------------------------------------------
        self.logging_util.start_logging()

        labeled_dicts = []
        for r in selected_rows:
            hp = self._row_dict(self.nn_train.nearestRow(r))
            score = self.model_wrapper.run_model(hp)
            fitness = 1 - score
            labeled_dicts.append(hp)
            self.logging_util.log(hp, fitness, 1)

        # ----------------------------------------------
        # 3. Build sorter model from labeled rows
        # ----------------------------------------------
        sorter = self._build_sorter(labeled_dicts)

        # ----------------------------------------------
        # 4. Evaluate FULL TEST SET just like ActLearn
        # ----------------------------------------------
        test_rows = self.X_test.values.tolist()

        test_dicts = [
            self._row_dict(self.nn_test.nearestRow(row))
            for row in test_rows
        ]

        # Sort test rows by the surrogate model
        test_sorted = sorted(test_dicts, key=sorter)

        # ----------------------------------------------
        # 5. Pick best test row using TRUE evaluator
        # ----------------------------------------------
        best_hp = None
        best_fitness = float("inf")

        for hp in test_sorted:
            score = self.model_wrapper.run_model(hp)
            fitness = 1 - score
            if fitness < best_fitness:
                best_fitness = fitness
                best_hp = hp

        self.logging_util.stop_logging()

        self.best_config = best_hp
        self.best_value = best_fitness

        print(f"Best config = {self.best_config}")
        print(f"Best value  = {self.best_value}")

        return self.best_config, self.best_value
