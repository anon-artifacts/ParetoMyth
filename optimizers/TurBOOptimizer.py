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


class TurBOOptimizer(BaseOptimizer):
    """
    TurBO optimizer following the same structure as TPEOptimizer:

    ✔ Uses discrete ConfigSpace
    ✔ Suggests per-dimension values
    ✔ Projects to nearest valid dataset row
    ✔ Evaluates model + caching
    """

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)

        self.nn = Data(
            self.X_df.values.tolist(),
            column_types=self.model_config.column_types,
        )

        self.config_space, self.param_names, _ = self.model_config.get_configspace()

        self.cache = {}
        self.best_config = None
        self.best_value = None

    # ------------ Helpers ------------ #

    def _clean(self, v):
        return v.item() if hasattr(v, "item") else v

    def _nearest_row(self, hp_dict):
        query = [hp_dict[c] for c in self.columns]
        row = self.nn.nearestRow(query)
        return {c: self._clean(v) for c, v in zip(self.columns, row)}

    def _row_tuple(self, hp_dict):
        return tuple(hp_dict[c] for c in self.columns)

    # ------------ Objective ------------ #

    def _objective(self, trial: optuna.Trial):
        raw_hp = {}

        for hp in self.config_space.get_hyperparameters():

            if isinstance(hp, Constant):
                raw_hp[hp.name] = hp.value

            elif isinstance(hp, OrdinalHyperparameter):
                raw_hp[hp.name] = trial.suggest_categorical(hp.name, list(hp.sequence))

            elif isinstance(hp, CategoricalHyperparameter):
                raw_hp[hp.name] = trial.suggest_categorical(hp.name, list(hp.choices))

            else:
                raise ValueError(f"Unsupported hyperparameter type {type(hp)}")

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

    # ------------ Public API ------------ #

    def optimize(self):
        n_trials = self.config["n_trials"]
        self.logging_util.start_logging()
        dim = len(self.param_names)
        # TurBO sampler
        sampler = optunahub.load_module("samplers/turbo").TuRBOSampler(seed=self.seed, deterministic_objective=True, failure_tolerance=max(5, dim))

        study_name = f"turbo_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:4]}"
        study = optuna.create_study(direction="minimize", sampler=sampler,
                                    study_name=study_name)

        study.optimize(self._objective, n_trials=n_trials)

        # Collect final best parameters
        final_raw = dict(study.best_params)

        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, Constant):
                final_raw[hp.name] = hp.value

        final_raw = {k: self._clean(v) for k, v in final_raw.items()}
        final_hp = self._nearest_row(final_raw)
        final_score = 1 - self.model_wrapper.run_model(final_hp)

        self.best_config = final_hp
        self.best_value = final_score

        self.logging_util.stop_logging()
        return self.best_config, self.best_value
