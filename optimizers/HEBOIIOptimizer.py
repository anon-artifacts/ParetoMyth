import optuna
import optunahub
import time
import uuid

from optimizers.base_optimizer import BaseOptimizer
from models.Data import Data
from ConfigSpace.hyperparameters import (
    OrdinalHyperparameter,
    CategoricalHyperparameter,
    Constant,
)


class HEBOOptimizer(BaseOptimizer):
    """
    HEBO optimizer following the same structure as TPE / TurBO:

    ✔ Uses discrete ConfigSpace
    ✔ Samples encoded hyperparameters
    ✔ Projects to nearest dataset row using KD-tree
    ✔ Uses caching and logging_util
    ✔ Returns best_config, best_value
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        # Encoded design matrix
        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)

        # KD-tree for nearest discrete row
        self.nn = Data(
            self.X_df.values.tolist(),
            column_types=self.model_config.column_types,
        )

        # ConfigSpace definition
        self.config_space, self.param_names, _ = self.model_config.get_configspace()

        self.cache = {}
        self.best_config = None
        self.best_value = None

    # --------------------- Helpers --------------------- #

    def _clean(self, v):
        """Convert numpy scalars → Python types."""
        return v.item() if hasattr(v, "item") else v

    def _nearest_row(self, hp_dict):
        """Project hyperparameter dict onto nearest encoded row."""
        query = [hp_dict[c] for c in self.columns]
        row = self.nn.nearestRow(query)
        return {c: self._clean(v) for c, v in zip(self.columns, row)}

    def _row_tuple(self, hp_dict):
        """Tuple key for caching."""
        return tuple(hp_dict[c] for c in self.columns)

    # ------------------- Optuna Objective ------------------- #

    def _objective(self, trial: optuna.Trial):
        raw_hp = {}

        # Suggest values according to ConfigSpace
        for hp in self.config_space.get_hyperparameters():

            if isinstance(hp, Constant):
                raw_hp[hp.name] = hp.value

            elif isinstance(hp, OrdinalHyperparameter):
                raw_hp[hp.name] = trial.suggest_categorical(hp.name, list(hp.sequence))

            elif isinstance(hp, CategoricalHyperparameter):
                raw_hp[hp.name] = trial.suggest_categorical(hp.name, list(hp.choices))

            else:
                raise ValueError(f"Unsupported hyperparameter type: {type(hp)}")

        # Project into nearest row
        valid_hp = self._nearest_row(raw_hp)
        key = self._row_tuple(valid_hp)

        # Caching
        if key in self.cache:
            fitness = self.cache[key]
        else:
            score = self.model_wrapper.run_model(valid_hp)
            fitness = 1 - score
            self.cache[key] = fitness

        # Logging
        self.logging_util.log(valid_hp, fitness, 1)
        return fitness

    # ------------------- Optimize ------------------- #

    def optimize(self):
        if not self.logging_util:
            raise ValueError("Logging util not provided.")

        n_trials = self.config["n_trials"]
        self.logging_util.start_logging()

        # Load HEBO sampler
        module = optunahub.load_module("samplers/hebo")
        sampler = module.HEBOSampler(seed=self.seed)

        study_name = f"hebo_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:4]}"
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name=study_name,
        )

        study.optimize(self._objective, n_trials=n_trials)

        # Build final hyperparameter set
        final_raw = {}

        # 1) Add parameters discovered by HEBO
        for name, value in study.best_params.items():
            final_raw[name] = self._clean(value)

        # 2) Add constants from ConfigSpace
        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, Constant):
                final_raw[hp.name] = hp.value

        final_raw = {k: self._clean(v) for k, v in final_raw.items()}

        # Final projection to real dataset row
        final_hp = self._nearest_row(final_raw)
        final_score = 1 - self.model_wrapper.run_model(final_hp)

        self.best_config = final_hp
        self.best_value = final_score

        self.logging_util.stop_logging()
        return final_hp, final_score
