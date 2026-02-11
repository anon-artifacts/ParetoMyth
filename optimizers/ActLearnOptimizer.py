from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import active_learning.src.bl as bl
from optimizers.base_optimizer import BaseOptimizer


class ActLearnOptimizer(BaseOptimizer):
    """
    Pure BL Active Learning optimizer.
    Bypasses ModelWrapper entirely—no encoding, no lookup.
    Uses BL's own ydist as the objective score.
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        self.best_config = None
        self.best_value = None

        # Load BL dataset immediately (so optimize() time is pure AL time)
        csv_path = model_config.get_dataset_file()
        self.bl_data = bl.Data(bl.csv(csv_path))

        if len(self.bl_data.cols.y) == 0:
            raise ValueError("BL could not detect Y columns in dataset CSV.")


    def optimize(self):
        n_trials = self.config["n_trials"]
        print(f"=== Running BL Active Learning (budget={n_trials}) ===")

        # 1. Set BL stopping condition (new runs start fresh)
        bl.the.Stop = n_trials

        # 2. Run BL ActiveLearner
        result = bl.actLearn(self.bl_data, shuffle=True)

        # 3. Best row from BL
        best_row = bl.first(result.best.rows)

        # 4. Extract only X columns
        x_len = len(self.bl_data.cols.x)
        best_hp = dict(zip(self.bl_data.cols.names[:x_len], best_row[:x_len]))

        # 5. Evaluate using BL’s internal distance-to-heaven metric
        best_value = bl.ydist(best_row, self.bl_data)

        # 6. Logging
        self.logging_util.start_logging()
        self.logging_util.log(best_hp, best_value, 1)
        self.logging_util.stop_logging()

        self.best_config = best_hp
        self.best_value = best_value

        print(f"Best config = {self.best_config}")
        print(f"Best value  = {self.best_value}")

        return self.best_config, self.best_value
