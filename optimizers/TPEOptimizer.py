import optuna
import numpy as np
import time
import uuid

from optimizers.base_optimizer import BaseOptimizer
from models.Data import Data
from ConfigSpace.hyperparameters import (
    OrdinalHyperparameter,
    CategoricalHyperparameter,
    Constant,
)


class TPEOptimizer(BaseOptimizer):
    """
    Optuna TPE optimizer that matches the SMAC/ActLearn pattern:

    ✔ Uses ModelConfigurationStatic's discrete ConfigSpace
    ✔ Samples each hyperparameter independently (TPE sees full structure)
    ✔ Values are the *encoded* discrete choices from the tabular data
    ✔ Maps sampled config -> nearest valid row via KD-tree (Data)
    ✔ Evaluates model_wrapper.run_model on that nearest row
    ✔ Uses caching + logging_util
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        # Encoded design matrix
        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)

        # KD-tree / NN helper on encoded rows
        self.nn = Data(
            self.X_df.values.tolist(),
            column_types=self.model_config.column_types,
        )

        # Discrete ConfigSpace from ModelConfigurationStatic
        self.config_space, self.param_names, _ = self.model_config.get_configspace()

        self.cache = {}
        self.best_config = None
        self.best_value = None

    # ---------------- Helpers ---------------- #

    def _clean(self, v):
        """Convert numpy scalars to native Python types."""
        if hasattr(v, "item"):
            return v.item()
        return v

    def _nearest_row(self, hp_dict):
        """Project hyperparameter dict onto nearest existing row in the table."""
        print(hp_dict)
        query = [hp_dict[col] for col in self.columns]
        row = self.nn.nearestRow(query)
        return {c: self._clean(v) for c, v in zip(self.columns, row)}

    def _row_tuple(self, hp_dict):
        """Tuple key for caching."""
        return tuple(hp_dict[c] for c in self.columns)

    # ---------------- Optuna Objective ---------------- #

    def _objective(self, trial: optuna.Trial):
        """
        One TPE trial:

        1. Sample a discrete hyperparameter configuration from ConfigSpace
           using Optuna's suggest_categorical per dimension.
        2. Map to nearest existing row in the encoded dataset.
        3. Evaluate model_wrapper.run_model on that row.
        4. Log + cache + return fitness (1 - score).
        """
        raw_hp = {}

        for hp in self.config_space.get_hyperparameters():
            name = hp.name

            if isinstance(hp, Constant):
                raw_hp[name] = hp.value

            elif isinstance(hp, OrdinalHyperparameter):
                # Values are already encoded & sorted in ModelConfigurationStatic
                choices = list(hp.sequence)
                raw_hp[name] = trial.suggest_categorical(name, choices)

            elif isinstance(hp, CategoricalHyperparameter):
                choices = list(hp.choices)
                raw_hp[name] = trial.suggest_categorical(name, choices)

            else:
                raise ValueError(f"Unsupported hyperparameter type: {type(hp)}")

        # Map to nearest valid table row
        valid_hp = self._nearest_row(raw_hp)
        key = self._row_tuple(valid_hp)

        if key in self.cache:
            fitness = self.cache[key]
        else:
            score = self.model_wrapper.run_model(valid_hp)
            fitness = 1 - score
            self.cache[key] = fitness

        self.logging_util.log(valid_hp, fitness, 1)
        return fitness

    # ---------------- Public API ---------------- #

    def optimize(self):
        if not self.logging_util:
            raise ValueError("Logging util not provided.")

        n_trials = self.config["n_trials"]
        self.logging_util.start_logging()

        study_name = f"tpe_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:4]}"
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            study_name=study_name,
        )

        study.optimize(self._objective, n_trials=n_trials)

        # Best trial -> config
        best_params = study.best_params  # these are actual encoded values per param
       # Rebuild full hyperparameter dictionary including constants
        final_hp_raw = {}

        # 1. Add sampled parameters
        for name, value in study.best_params.items():
            final_hp_raw[name] = value

        # 2. Add constant hyperparameters
        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, Constant):
                final_hp_raw[hp.name] = hp.value   # IMPORTANT!!!
        final_hp_raw = {k: self._clean(v) for k, v in final_hp_raw.items()}

        # Final projection to the nearest table row (for consistency)
        final_hp = self._nearest_row(final_hp_raw)
        final_score = 1 - self.model_wrapper.run_model(final_hp)

        self.best_config = final_hp
        self.best_value = final_score

        self.logging_util.stop_logging()

        return self.best_config, self.best_value
