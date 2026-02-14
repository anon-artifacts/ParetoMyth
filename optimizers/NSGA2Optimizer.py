from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import optuna
from optuna.samplers import NSGAIISampler
from optimizers.base_optimizer import BaseOptimizer
from models.Data import Data
from ConfigSpace.hyperparameters import (
    OrdinalHyperparameter,
    CategoricalHyperparameter,
    Constant,
)
from utils import DistanceUtil
import time
import numpy as np


class NSGA2Optimizer(BaseOptimizer):
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
        self.num_objectives = len(
            self.model_wrapper.get_score(
                {c: self.X_df.iloc[0][c] for c in self.columns}
            )
        )
        self.iteration = 0
        self.population_size = 100

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
                raw_hp[hp.name] = trial.suggest_categorical(hp.name, list(hp.sequence))
            elif isinstance(hp, CategoricalHyperparameter):
                raw_hp[hp.name] = trial.suggest_categorical(hp.name, list(hp.choices))
            else:
                raise ValueError("Unsupported hyperparameter")

        valid_hp = self._nearest_row(raw_hp)
        key = self._row_tuple(valid_hp)

        if key in self.cache:
            scores = self.cache[key]
        else:
            try:
                scores = self.model_wrapper.get_score(valid_hp)
                scores = tuple(1 - s for s in scores)  # flip 1-d2h → d2h
            except Exception:
                scores = tuple(0.0 for _ in range(self.num_objectives))
            self.cache[key] = scores

        self.iteration += 1
        self.track_evaluation(valid_hp, scores, self.iteration)

        # At the end of each generation, count how many of THIS generation's
        # individuals are non-dominated against the full history so far.
        if self.iteration % self.population_size == 0:
            generation = self.iteration // self.population_size
            all_trials = [
                t for t in trial.study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]

            # Trials belonging to the current generation (0-indexed slice)
            gen_start = (generation - 1) * self.population_size
            gen_trials = all_trials[gen_start : gen_start + self.population_size]

            # A generation trial is non-dominated if no other trial (full history)
            # dominates it.
            gen_frontier = [
                t for t in gen_trials
                if not any(
                    self._is_dominated(t.values, other.values)
                    for other in all_trials
                    if other is not t
                )
            ]

            # frontier_size  = non-dominated individuals in this generation
            # total_population = this generation's size (always population_size)
            # frontier_percentage = fraction of this generation that survived
            self.track_frontier(generation, gen_frontier, self.population_size)

        return scores

    def _is_dominated(self, obj1, obj2):
        """Return True if obj1 is dominated by obj2 (obj2 is at least as good
        on all objectives and strictly better on at least one)."""
        at_least_as_good = all(b <= a for a, b in zip(obj1, obj2))
        strictly_better  = any(b <  a for a, b in zip(obj1, obj2))
        return at_least_as_good and strictly_better

    def optimize(self):
        n_trials = self.config["n_trials"]

        self.start_time = time.time()

        sampler = NSGAIISampler(
            population_size=self.population_size,
            seed=self.seed,
        )

        study = optuna.create_study(
            directions=["minimize"] * self.num_objectives,
            sampler=sampler,
        )

        study.optimize(self._objective, n_trials=n_trials, catch=(Exception,))
        # No second track_frontier call here — every generation was already
        # recorded inside _objective, including the final one.

        # Final frontier: all non-dominated trials across the whole study
        all_trials = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        frontier = [
            t for t in all_trials
            if not any(
                self._is_dominated(t.values, other.values)
                for other in all_trials
                if other is not t
            )
        ]

        # Best solution by d2h
        best_trial = None
        best_d2h_norm = float("inf")
        ideal = [0] * self.num_objectives

        for t in frontier:
            d2h_norm = DistanceUtil.d2h(ideal, t.values)
            if d2h_norm < best_d2h_norm:
                best_d2h_norm = d2h_norm
                best_trial = t

        best_raw = dict(best_trial.params)
        for hp in self.config_space.get_hyperparameters():
            if isinstance(hp, Constant):
                best_raw[hp.name] = hp.value

        final_hp = self._nearest_row(best_raw)

        self.best_config = final_hp
        self.best_value = best_d2h_norm
        self.end_time = time.time()

        return final_hp, self.best_value
