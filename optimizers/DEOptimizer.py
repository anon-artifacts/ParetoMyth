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


class DEOptimizer(BaseOptimizer):

    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)

        self.X_df = self.model_wrapper.X
        self.columns = list(self.X_df.columns)

        self.nn = Data(
            self.X_df.values.tolist(),
            column_types=self.model_config.column_types,
        )

        self.config_space, _, _ = self.model_config.get_configspace()
        self.cache = {}

    def _clean(self, v):
        return v.item() if hasattr(v, "item") else v

    def _nearest_row(self, hp_dict):
        query = [hp_dict[c] for c in self.columns]
        row = self.nn.nearestRow(query)
        return {c: self._clean(v) for c, v in zip(self.columns, row)}

    def _row_tuple(self, hp_dict):
        return tuple(hp_dict[c] for c in self.columns)

    def _objective(self, trial):
        raw_hp = {}

        for hp in self.config_space.get_hyperparameters():

            if isinstance(hp, Constant):
                raw_hp[hp.name] = hp.value

            elif isinstance(hp, OrdinalHyperparameter):
                raw_hp[hp.name] = trial.suggest_float(hp.name, 0, len(hp.sequence) - 1)

            elif isinstance(hp, CategoricalHyperparameter):
                raw_hp[hp.name] = trial.suggest_float(hp.name, 0, len(hp.choices) - 1)

            else:
                raise ValueError("Unsupported hyperparameter")

        valid_hp = self._nearest_row(raw_hp)
        key = self._row_tuple(valid_hp)

        if key in self.cache:
            fitness = self.cache[key]
        else:
            score = self.model_wrapper.run_model(valid_hp)
            fitness = 1 - score
            self.cache[key] = fitness

        return fitness

    def optimize(self):
        n_trials = self.config["n_trials"]

        sampler_module = optunahub.load_module("samplers/differential_evolution")
        sampler = sampler_module.DESampler(seed=self.seed, population_size="auto")

        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(self._objective, n_trials=n_trials)

        # Process best solution
        final_hp_raw = dict(study.best_params)
        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, Constant):
                final_hp_raw[hp.name] = hp.value

        final_hp = self._nearest_row(final_hp_raw)
        final_score = 1 - self.model_wrapper.run_model(final_hp)

        self.best_config = final_hp
        self.best_value = final_score

        return final_hp, final_score