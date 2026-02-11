from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import active_learning.src.bl as bl
from optimizers.base_optimizer import BaseOptimizer


class SWAYOptimizer(BaseOptimizer):
    """
    Pure BL Branch/SWAY-style optimizer.
    - Loads BL Data from the dataset CSV
    - Runs BL's branch() (recursive halving via twoFar/half projection)
    - Picks the best row (min ydist) from the final branch subset
    - Extracts X columns as the config and logs once
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        self.best_config = None
        self.best_value = None

        csv_path = model_config.get_dataset_file()
        self.bl_data = bl.Data(bl.csv(csv_path))

        if len(self.bl_data.cols.y) == 0:
            raise ValueError("BL could not detect Y columns in dataset CSV.")

    def optimize(self):
        # Use n_trials as branch depth unless you prefer log2 scaling (see note below)
        depth = int(self.config.get("n_trials", 20))

        # Optional knobs (can be placed in config if you want)
        far = float(self.config.get("far", 0.90))      # like Lua the.far
        sortp = bool(self.config.get("sortp", True))   # orient endpoints by ydist

        print(f"=== Running BL Branch (depth={depth}, far={far}, sortp={sortp}) ===")

        # Seed control (BL uses global random; ensure determinism per run)
        bl.random.seed(self.seed)

        # Run branch on a shuffled copy of rows to match Lua behavior
        rows = list(self.bl_data.rows)
        bl.random.shuffle(rows)

        # Run BL branch: returns a subset of rows
        subset = bl.branch(self.bl_data, budget=depth, rows=rows, sortp=sortp, far=far)

        if not subset:
            raise RuntimeError("BL branch returned empty subset")

        # Pick best row by BL's own objective (ydist)
        best_row = min(subset, key=lambda r: bl.ydist(r, self.bl_data))

        # Extract only X columns as hp config
        x_len = len(self.bl_data.cols.x)
        best_hp = dict(zip(self.bl_data.cols.names[:x_len], best_row[:x_len]))

        # Objective value
        best_value = bl.ydist(best_row, self.bl_data)

        # Logging (single evaluation)
        self.logging_util.start_logging()
        self.logging_util.log(best_hp, best_value, 1)
        self.logging_util.stop_logging()

        self.best_config = best_hp
        self.best_value = best_value

        print(f"Best config = {self.best_config}")
        print(f"Best value  = {self.best_value}")

        return self.best_config, self.best_value
