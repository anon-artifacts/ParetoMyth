from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
    Constant,
)
import numpy as np
import time
from optimizers.base_optimizer import BaseOptimizer
import random
import uuid
from sklearn.model_selection import train_test_split
from models.Data import Data
from ConfigSpace.configuration import Configuration
from smac import Scenario, HyperparameterOptimizationFacade as HPOFacade
from smac.utils.configspace import convert_configurations_to_array


class SMACOptimizer(BaseOptimizer):
    """
    ✔ Correct version for tabular HPO
    ✔ SMAC samples from the configspace only
    ✔ Each sample is mapped to the nearest row in the tabular dataset
    ✔ All evaluations come from the table
    ✔ No train/test split
    ✔ No surrogate prediction on held-out data
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)
        self.best_config = None
        self.best_value = None        
        self.cache = {}

        # SMAC configspace
        self.config_space, _, _ = self.model_config.get_configspace()
        # ---- Detect mode ----
        # Tabular wrappers in your code have X
        self.is_tabular = hasattr(self.model_wrapper, "X")

        if self.is_tabular:
            # Tabular encoded dataset
            self.X_df = self.model_wrapper.X
            self.columns = list(self.X_df.columns)

            # KD-tree / NN helper (your Data class)
            self.nn = Data(
                self.X_df.values.tolist(),
                column_types=self.model_config.column_types)

        else:
            # Analytical (DTLZ): param_names should exist and match configspace names
            self.columns = list(self.model_config.param_names)

    def _clean(self, v):
        if isinstance(v, (np.integer, np.int64)): return int(v)
        if isinstance(v, (np.floating, np.float64)): return float(v)
        return v

    def _config_to_dict(self, config: Configuration):
        return {p: self._clean(config[p]) for p in self.model_config.param_names}

    def _row_tuple(self, hyperparams):
        return tuple(hyperparams[p] for p in self.model_config.param_names)

    def _nearest_row(self, hyperparams_dict):
        """
        Tabular: snap to nearest row in dataset.
        Analytical: identity mapping (use as-is).
        """
        if not self.is_tabular:
            return hyperparams_dict
        query = [hyperparams_dict[col] for col in self.columns]
        nn_row = self.nn.nearestRow(query)

        return {col: self._clean(v) for col, v in zip(self.columns, nn_row)}

    def optimize(self):

        n_trials = self.config["n_trials"]
        total_budget = self.config["n_trials"]

        # ------------------------------------------------#
        # Define SMAC Objective
        # ------------------------------------------------#
        def objective(config: Configuration, seed=0):
            raw_hp = self._config_to_dict(config)
            valid_hp = self._nearest_row(raw_hp)

            key = self._row_tuple(valid_hp)
            if key in self.cache:
                return self.cache[key]

            score = self.model_wrapper.run_model(valid_hp)
            fitness = 1 - score

            self.cache[key] = fitness

            return fitness

        output_directory = (
            f"{self.config['output_directory']}/"
            f"smac_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:4]}"
        )

        # Scenario
        scenario = Scenario(
            configspace=self.config_space,
            n_trials=n_trials,
            deterministic=True,
            output_directory=output_directory,
            seed=self.seed,
        )

        # SMAC driver
        initial_design = HPOFacade.get_initial_design(scenario)

        smac = HPOFacade(
            scenario=scenario,
            target_function=objective,
            initial_design=initial_design,
            overwrite=True,
        )

        try:
            incumbent = smac.optimize()
        except Exception:
            incumbent = smac.optimizer.intensifier.get_incumbent()

        final_dict = self._nearest_row(self._config_to_dict(incumbent))
        final_score = 1 - self.model_wrapper.run_model(final_dict)

        self.best_config = final_dict
        self.best_value = final_score

        return self.best_config, self.best_value
