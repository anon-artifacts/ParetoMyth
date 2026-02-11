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
    def __init__(self, config, model_wrapper, model_config, logging_util, seed):
        super().__init__(config, model_wrapper, model_config, logging_util, seed)
        self.best_config = None
        self.best_value = None
        
        # Get the ENCODED dataframe from model_wrapper
        self.X_df = self.model_wrapper.X_encoded
        self.columns = list(self.X_df.columns)
        self.cache = {}
        
        # ------------------------------------------------------
        # 1. Train/Test split on ENCODED data
        # ------------------------------------------------------
        self.X_train, self.X_test = train_test_split(
            self.X_df,
            test_size=0.5,
            random_state=self.seed,
        )
        
        # ------------------------------------------------------
        # 2. KD-trees with ENCODED data
        # ------------------------------------------------------
        self.nn_train = Data(
            self.X_train.values.tolist(),
            column_types=self.model_config.column_types
        )
        self.nn_test = Data(
            self.X_test.values.tolist(),
            column_types=self.model_config.column_types
        )
        
        # ------------------------------------------------------
        # 3. ConfigSpace
        # ------------------------------------------------------
        self.config_space, _, _ = self.model_config.get_configspace()
    
    def set_model_config(self, model_config):
        """Override to initialize ConfigSpace when model_config is set."""
        super().set_model_config(model_config)
        if self.config_space is None:
            self.config_space, _, _ = self.model_config.get_configspace()
    
    def _clean(self, v):
        if isinstance(v, (np.integer, np.int64)):
            return int(v)
        if isinstance(v, (np.floating, np.float64)):
            return float(v)
        return v
    
    def _config_to_dict(self, config: Configuration):
        """Convert SMAC config to dict - values are already in correct encoding"""
        return {p: self._clean(config[p]) for p in self.model_config.param_names}
    
    def _row_tuple(self, hyperparams: dict):
        return tuple(hyperparams[p] for p in self.model_config.param_names)
    
    def _valid_row_train(self, hyperparams: dict):
        """NN fallback using TRAIN KD-tree - input is already encoded"""
        q = [hyperparams[col] for col in self.columns]
        nn_row = self.nn_train.nearestRow(q)
        return {col: self._clean(v) for col, v in zip(self.columns, nn_row)}
    
    def _valid_row_test(self, hyperparams: dict):
        """NN fallback using TEST KD-tree - input is already encoded"""
        q = [hyperparams[col] for col in self.columns]
        nn_row = self.nn_test.nearestRow(q)
        return {col: self._clean(v) for col, v in zip(self.columns, nn_row)}
    
    def optimize(self):
        """Run SMAC optimization"""
        if not self.logging_util:
            raise ValueError("Logging utility not set!")
        
        print("Starting SMAC optimization (using encoded data)")
        total_budget = self.config["n_trials"]
        
        def objective(config: Configuration, seed: int = 0):
            # Config values are already properly encoded (dates as floats, etc.)
            raw_hp = self._config_to_dict(config)
            eval_hp = self._valid_row_train(raw_hp)
            
            print(f"RAW HP (from SMAC): {raw_hp}")
            print(f"EVAL HP (after NN): {eval_hp}")
            
            key = self._row_tuple(eval_hp)
            if key in self.cache:
                return self.cache[key]
            
            score = self.model_wrapper.run_model(eval_hp)
            fitness = 1 - score
            self.logging_util.log(eval_hp, fitness, 1)
            self.cache[key] = fitness
            return fitness

        def eval_test(config: Configuration, seed: int =0):
            raw_hp = self._config_to_dict(config)
            eval_hp = self._valid_row_test(raw_hp)
            score = self.model_wrapper.run_model(eval_hp)
            fitness = 1 - score
            self.logging_util.log(eval_hp, fitness, 1)
            return fitness
        output_directory = (
            f"{self.config['output_directory']}/"
            f"smac_{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:4]}"
        )
        
        scenario = Scenario(
            configspace=self.config_space,
            n_trials=total_budget,
            deterministic=True,
            output_directory=output_directory,
            seed=self.seed,
        )
        
        initial_design = HPOFacade.get_initial_design(
            scenario,
            n_configs=max(1, int(total_budget * 0.1)),
        )
        
        smac = HPOFacade(
            scenario=scenario,
            target_function=objective,
            initial_design=initial_design,
            overwrite=True,
        )
        
        self.logging_util.start_logging()
        try:
            incumbent = smac.optimize()
        except Exception:
            incumbent = smac.optimizer.intensifier.get_incumbent()
        
        # Evaluate on test set
        model = smac._model
        
        # Create Configuration objects from encoded test data
        test_configs = []
        test_dicts = []
        for _, row in self.X_test.iterrows():
            hp = {c: self._clean(row[c]) for c in self.columns}
            test_dicts.append(hp)
            try:
                config_obj = Configuration(self.config_space, values=hp)
                test_configs.append(config_obj)
            except Exception as e:
                print(f"Warning: Could not create config for {hp}: {e}")
                test_configs.append(None)
        
        valid_test_configs = [c for c in test_configs if c is not None]
        valid_test_dicts = [d for c, d in zip(test_configs, test_dicts) if c is not None]
        
        if not valid_test_configs:
            print("Warning: No valid test configurations, using incumbent")
            self.best_config = None
            self.best_value = None
        else:
            test_vectors = convert_configurations_to_array(valid_test_configs)
            try:
                mu, _ = model.predict(test_vectors)
                best_idx = int(np.argmin(mu))
                best_hp = valid_test_dicts[best_idx]
                best_hp = self._valid_row_test(best_hp)
                
                self.best_config = best_hp
                self.best_value = eval_test(Configuration(self.config_space, values=best_hp))
            except Exception as e:
                print(f"Warning: Model prediction failed: {e}")
                self.best_config = None
                self.best_value = None
        
        self.logging_util.stop_logging()
        return self.best_config, self.best_value