# optimizers/MOTPEOptimizer.py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import optuna
from optuna.samplers import TPESampler
from optimizers.base_optimizer import BaseOptimizer
from models.Data import Data
from ConfigSpace.hyperparameters import (
    OrdinalHyperparameter,
    CategoricalHyperparameter,
    Constant,
)
from utils import DistanceUtil
import numpy as np
import time

class MOTPEOptimizer(BaseOptimizer):
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
        
        # Track evaluation
        self.iteration += 1
        self.track_evaluation(valid_hp, scores, self.iteration)
        
        return scores
    
    def _is_dominated(self, obj1, obj2):
        """Check if obj1 is dominated by obj2 (minimization)"""
        better_in_any = False
        for a, b in zip(obj1, obj2):
            if a > b:
                better_in_any = True
            elif a < b:
                return False
        return better_in_any
    
    def _get_pareto_frontier(self, trials):
        """Extract Pareto frontier from trials"""
        frontier = []
        for trial in trials:
            is_dominated = False
            for other in trials:
                if trial != other and self._is_dominated(trial.values, other.values):
                    is_dominated = True
                    break
            if not is_dominated:
                frontier.append(trial)
        return frontier
    
    def optimize(self):
        n_trials = self.config["n_trials"]
        self.start_time = time.time()
        
        sampler = TPESampler(seed=self.seed,multivariate=True)
        study = optuna.create_study(
            directions=["minimize"] * self.num_objectives,
            sampler=sampler,
        )
        
        study.optimize(self._objective, n_trials=n_trials, catch=(Exception,))
        
        # Track frontier at end
        all_trials = study.trials
        frontier = self._get_pareto_frontier(all_trials)
        self.track_frontier(n_trials, frontier, len(all_trials))
        
        # Find best by d2h
        best_trial = None
        best_d2h = float("inf")
        ideal = [0] * self.num_objectives  # Minimization
        
        for t in frontier:
            d2h = DistanceUtil.d2h(ideal, t.values)
            d2h_norm = d2h  #this is because it is already normalized
            if d2h < best_d2h:
                best_d2h = d2h
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