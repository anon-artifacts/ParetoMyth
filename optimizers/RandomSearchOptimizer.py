import random
import numpy as np
from optimizers.base_optimizer import BaseOptimizer
from models.Data import Data


class RandomSearchOptimizer(BaseOptimizer):
    """
    Pure RANDOM SEARCH:
    - Samples random rows from the FULL table
    - Evaluates the model on each sampled row
    - Returns the best configuration found
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        random.seed(seed)
        np.random.seed(seed)

        self.best_config = None
        self.best_value = None

        # FULL encoded dataframe
        self.X_df = self.model_wrapper.X_encoded
        self.columns = list(self.X_df.columns)

        # KD-tree for nearest-row matching
        self.nn = Data(
            self.X_df.values.tolist(),
            column_types=self.model_config.column_types
        )

    # ---------------------------------------------------
    # Helpers
    # ---------------------------------------------------
    def _clean(self, v):
        import numpy as np
        if isinstance(v, (np.integer, np.int64)): 
            return int(v)
        if isinstance(v, (np.floating, np.float64)): 
            return float(v)
        return v

    def _row_dict(self, row):
        return {col: self._clean(v) for col, v in zip(self.columns, row)}

    # ---------------------------------------------------
    # RANDOM SEARCH
    # ---------------------------------------------------
    def optimize(self):
        budget = self.config["n_trials"]
        print(f"=== Running RANDOM SEARCH optimizer, budget={budget} ===")

        # All rows from encoded table
        rows = self.X_df.values.tolist()
        n = len(rows)

        self.logging_util.start_logging()

        best_hp = None
        best_fitness = float("inf")

        for _ in range(budget):
            # 1. Pick a random row
            rnd_row = rows[random.randint(0, n - 1)]

            # 2. Convert to nearest valid row (ensures stable encoding)
            hp = self._row_dict(self.nn.nearestRow(rnd_row))

            # 3. Evaluate with the model
            score = self.model_wrapper.run_model(hp)
            fitness = 1 - score

            # 4. Logging
            self.logging_util.log(hp, fitness, 1)

            # 5. Track best result
            if fitness < best_fitness:
                best_fitness = fitness
                best_hp = hp

        self.logging_util.stop_logging()

        self.best_config = best_hp
        self.best_value  = best_fitness

        print(f"Best config = {self.best_config}")
        print(f"Best value  = {self.best_value}")

        return self.best_config, self.best_value
