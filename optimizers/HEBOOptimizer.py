import time
import uuid
import numpy as np
import pandas as pd

from optimizers.base_optimizer import BaseOptimizer
from models.Data import Data

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

import ConfigSpace as CS
from ConfigSpace.hyperparameters import (
    OrdinalHyperparameter,
    CategoricalHyperparameter,
    Constant,
)


class HEBOOptimizer(BaseOptimizer):
    """
    HEBO optimizer for SEOptBench:

    ✔ Uses *your* ConfigSpace (ModelConfigurationStatic)
    ✔ Converts ConfigSpace → HEBO DesignSpace
    ✔ Dynamic warm-up: warmup = 10% of n_trials (at least 1)
    ✔ No off-table values (all categorical)
    ✔ Nearest-row evaluation identical to SMAC/TPE
    ✔ Fully deterministic stopping after n_trials
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        # Encoded dataset
        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)

        # KD-tree nearest row lookup
        self.nn = Data(
            self.X_df.values.tolist(),
            column_types=self.model_config.column_types
        )

        # ConfigSpace (from ModelConfigurationStatic)
        self.configspace, self.param_names, _ = self.model_config.get_configspace()

        # Convert CS → HEBO space
        self.design_space = self._convert_configspace_to_hebo(self.configspace)

        # Initialize cache, results
        self.cache = {}
        self.best_config = None
        self.best_value = None

        # Trials budget
        self.n_trials = self.config["n_trials"]

    # ----------------------------------------------------------
    # Convert ConfigSpace → HEBO DesignSpace
    # ----------------------------------------------------------
    def _convert_configspace_to_hebo(self, cs: CS.ConfigurationSpace):
        space_cfg = []

        for hp in cs.get_hyperparameters():

            if isinstance(hp, Constant):
                space_cfg.append({
                    "name": hp.name,
                    "type": "cat",
                    "categories": [hp.value],
                })

            elif isinstance(hp, CategoricalHyperparameter):
                space_cfg.append({
                    "name": hp.name,
                    "type": "cat",
                    "categories": list(hp.choices),
                })

            elif isinstance(hp, OrdinalHyperparameter):
                space_cfg.append({
                    "name": hp.name,
                    "type": "cat",
                    "categories": list(hp.sequence),
                })

            else:
                raise ValueError(f"Unsupported hyperparameter type: {type(hp)}")

        return DesignSpace().parse(space_cfg)

    # ----------------------------------------------------------
    # Utility functions
    # ----------------------------------------------------------
    def _clean(self, v):
        if hasattr(v, "item"):
            return v.item()
        return v

    def _nearest_row(self, hp):
        query = [hp[c] for c in self.columns]
        row = self.nn.nearestRow(query)
        return {c: self._clean(v) for c, v in zip(self.columns, row)}

    def _row_key(self, hp):
        return tuple(hp[c] for c in self.columns)

    def _evaluate(self, hp):
        key = self._row_key(hp)
        if key in self.cache:
            return self.cache[key]

        score = self.model_wrapper.run_model(hp)
        fitness = 1 - score  # minimize

        self.logging_util.log(hp, fitness, 1)
        self.cache[key] = fitness
        return fitness

    # ----------------------------------------------------------
    # Main HEBO optimization loop
    # ----------------------------------------------------------
    def optimize(self):
        if not self.logging_util:
            raise ValueError("logging_util must be provided.")

        self.logging_util.start_logging()

        # Dynamic warm-up = 10% of n_trials (at least 1)
        warmup = max(1, min(self.n_trials, self.n_trials // 10))

        opt = HEBO(
            model_name="gp",
            space=self.design_space,
            rand_sample=warmup
        )

        # Manual loop ensures EXACT stopping
        for i in range(self.n_trials):
            # HEBO suggests 1 point
            rec_df = opt.suggest(n_suggestions=1)
            rec = rec_df.iloc[0].to_dict()

            # Clean + nearest-row projection
            raw_hp = {k: self._clean(v) for k, v in rec.items()}
            valid_hp = self._nearest_row(raw_hp)

            # Evaluate
            fitness = self._evaluate(valid_hp)

            # Report result to HEBO
            opt.observe(rec_df, np.array([[fitness]], dtype=float))

        # Select best from HEBO's record
        best_idx = int(opt.y.argmin())
        best_raw = opt.X.iloc[best_idx].to_dict()

        # Final projection to table
        best_hp = self._nearest_row({k: self._clean(v) for k, v in best_raw.items()})
        best_score = 1 - self.model_wrapper.run_model(best_hp)

        self.best_config = best_hp
        self.best_value = best_score

        self.logging_util.stop_logging()
        return self.best_config, self.best_value
