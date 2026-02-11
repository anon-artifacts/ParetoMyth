from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import tempfile
import numpy as np

import active_learning.src.bl as bl
from optimizers.base_optimizer import BaseOptimizer


class ActLearnOptimizer(BaseOptimizer):
    """
    BL Active Learning optimizer.

    Supports:
    - Tabular mode: uses existing dataset CSV.
    - DTLZ mode: builds a synthetic pool CSV (numeric X vars + Score+),
      then runs BL actLearn over that pool.

    IMPORTANT:
    - Does NOT modify bl.py
    - Uses BL's uppercase-first-letter rule so Y columns become Num (have .goal)
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        self.best_config = None
        self.best_value = None
        self.seed = seed

        self.pool_size = int(self.config.get("pool_size", 5000))

        self.is_dtlz = (
            hasattr(model_config, "problem") and
            hasattr(model_config, "n_vars") and
            hasattr(model_config, "n_objs") and
            hasattr(model_config, "param_names")
        )

        self.csv_path = None
        self.bl_data = None

        if not self.is_dtlz:
            self.csv_path = model_config.get_dataset_file()
            self.bl_data = bl.Data(bl.csv(self.csv_path))
            if len(self.bl_data.cols.y) == 0:
                raise ValueError("BL could not detect Y columns in dataset CSV.")

    # ------------------------------------------------------------------
    # DTLZ → BL pool construction
    # ------------------------------------------------------------------
    def _build_dtlz_pool_csv(self):
        """
        Create a temporary CSV for BL:

          X0,X1,...,X{n_var-1},Score+

        Notes:
        - Uppercase first letter forces Num columns in BL (so .goal exists).
        - Score+ means maximize, so goal=1.
        - score is expected in [0,1] (your wrapper returns 1 - d2h).
        """
        n_var = int(self.model_config.n_vars)
        param_names = list(self.model_config.param_names)

        rng = np.random.default_rng(self.seed)
        X = rng.random((self.pool_size, n_var))

        rows = []
        for i in range(self.pool_size):
            hp = {p: float(X[i, j]) for j, p in enumerate(param_names)}
            score = float(self.model_wrapper.run_model(hp))  # should be in [0,1]
            rows.append((X[i], score))

        fd, path = tempfile.mkstemp(prefix="dtlz_pool_", suffix=".csv")
        os.close(fd)

        with open(path, "w", encoding="utf-8", newline="") as f:
            # ✅ Uppercase column names so BL makes them Num (not Sym)
            header = [f"X{j}" for j in range(n_var)] + ["Score+"]
            f.write(",".join(header) + "\n")

            for xvec, score in rows:
                line = ",".join([f"{v:.10f}" for v in xvec] + [f"{score:.10f}"])
                f.write(line + "\n")

        # Safety check
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
            if not first.endswith("Score+"):
                raise RuntimeError(f"Bad BL header (must end with Score+): {first}")

        return path

    # ------------------------------------------------------------------
    # Main optimization
    # ------------------------------------------------------------------
    def optimize(self):
        if not self.logging_util:
            raise ValueError("Logging util not set")

        n_trials = int(self.config["n_trials"])
        print(f"=== Running BL Active Learning (budget={n_trials}) ===")

        bl.the.Stop = n_trials

        tmp_path = None
        if self.is_dtlz:
            tmp_path = self._build_dtlz_pool_csv()
            data = bl.Data(bl.csv(tmp_path))
        else:
            data = self.bl_data

        # (Optional) quick sanity: ensure y cols have goal
        for c in data.cols.y:
            if not hasattr(c, "goal"):
                raise RuntimeError(f"BL y-column missing goal (check header casing): {c.txt}")

        result = bl.actLearn(data, shuffle=True)
        best_row = bl.first(result.best.rows)

        # Build best config in YOUR framework's param_names (critical for DTLZ wrapper)
        if self.is_dtlz:
            # Use model_config.param_names order (x000.. etc in your framework)
            best_hp = {
                p: float(best_row[j])
                for j, p in enumerate(self.model_config.param_names)
            }
        else:
            # Tabular: extract X columns by BL's x cols length
            x_len = len(data.cols.x)
            x_names = data.cols.names[:x_len]
            best_hp = dict(zip(x_names, best_row[:x_len]))

        # BL ydist: lower is better in [0,1]
        best_ydist = float(bl.ydist(best_row, data))

        # Convert to "higher is better"
        best_value = 1.0 - best_ydist

        self.logging_util.start_logging()
        self.logging_util.log(best_hp, best_value, 1)
        self.logging_util.stop_logging()

        self.best_config = best_hp
        self.best_value = best_value

        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        print(f"Best config = {self.best_config}")
        print(f"Best value  = {self.best_value}")
        return self.best_config, self.best_value
