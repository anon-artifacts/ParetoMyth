from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import os
import glob
import time
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from models.configurations.model_config_static import ModelConfigurationStatic
from models.model_wrapper_static import ModelWrapperStatic

from optimizers.ActLearnOptimizerNoSplit import ActLearnOptimizer
from optimizers.DEHBOptimizer import DEHBOptimizer
from optimizers.SMACOptimizerNoSplit import SMACOptimizer
from optimizers.AroundOptimizerNoSplit import AroundOptimizer

from utils.LoggingUtil import LoggingUtil
from utils.data_loader_templated import load_data


def write_to_file(filepath, content):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)


def init_optimizer(optimizer_name, optimizer_config, model_wrapper, model_config, seed):
    optimizer_classes = {
        'DEHB': DEHBOptimizer,
        'Active_Learning': ActLearnOptimizer,
        'SMAC': SMACOptimizer,
        'KMPlusPlus': AroundOptimizer
    }
    if optimizer_name not in optimizer_classes:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer_classes[optimizer_name](
        optimizer_config, model_wrapper, model_config, None, seed
    )


def checkFileExists(path):
    return os.path.exists(path) and os.path.isfile(path)


def run_experiment(datasets, optimizers, repeats, checkpoints, tmp_output_dir, logging_dir):
    dataset_files = []

    if isinstance(datasets, str) and os.path.isdir(datasets):
        for root, _, _ in os.walk(datasets):
            dataset_files.extend(glob.glob(os.path.join(root, "*.csv")))
    else:
        dataset_files = [datasets] if isinstance(datasets, str) else datasets

    for dataset_file in dataset_files:
        optimize_single_dataset(
            optimizers, repeats, checkpoints,
            tmp_output_dir, logging_dir, dataset_file
        )


def run_repeat_wrapper(args):
    (logging_dir, data_name, model_wrapper, model_config, optimizer, checkpoint, optimizer_name, i) = args

    elapsed, best_cfg, best_val = run_single_repeat(
        logging_dir, data_name, model_wrapper, model_config,
        optimizer, checkpoint, optimizer_name, i
    )
    return str(best_cfg), str(best_val), str(elapsed)


def optimize_single_dataset(optimizers, repeats, checkpoints, tmp_output_dir, logging_dir, dataset_file):

    def terminate_all(signum, frame):
        print("Terminating all processes...")
        sys.exit(0)

    data_name = get_file_name(dataset_file)

    X, Y = load_data(dataset_file)
    hyperparameter_configs = {col: X[col].tolist() for col in X.columns}

   
    model_config = ModelConfigurationStatic(hyperparameter_configs, dataset_file, 1)
    model_wrapper = ModelWrapperStatic(X, Y, model_config)
    for optimizer in optimizers:
        if optimizer.get('disable'):
            continue

        for checkpoint in checkpoints:
            optimizer_name = optimizer['name']
            results_path = (os.path.join(tmp_output_dir, optimizer_name,
                                         f"{data_name}_{checkpoint}.csv")
                            if tmp_output_dir else None)

            if results_path and checkFileExists(results_path):
                continue

            optimizer['n_trials'] = checkpoint

            print(f"Running {optimizer_name}")
            results = {"configs": [], "best_values": [], "runtimes": []}

            args_list = [
                (logging_dir, data_name,
                 model_wrapper, model_config, optimizer, checkpoint, optimizer_name, i)
                for i in range(repeats)
            ]

            signal.signal(signal.SIGINT, terminate_all)

            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(run_repeat_wrapper, args)
                           for args in args_list]

                try:
                    for future in tqdm(as_completed(futures), total=len(futures),
                                       desc=f"{optimizer_name} {checkpoint}"):

                        best_cfg, best_val, elapsed = future.result()

                        results["configs"].append(best_cfg)
                        results["best_values"].append(best_val)
                        results["runtimes"].append(elapsed)

                except KeyboardInterrupt:
                    print("\nInterrupted. Shutting down...")
                    executor.shutdown(wait=False)
                    sys.exit(0)

            if results_path:
                content = "\n".join(
                    ", ".join(results[key]) for key in results
                )
                write_to_file(results_path, content)


def run_single_repeat(logging_dir, data_name, model_wrapper, model_config,
                      optimizer, checkpoint, optimizer_name, i):

    seed = i + 1
    log_filename = os.path.join(logging_dir, optimizer_name, f"{data_name}_{seed}.csv")
    model_config.set_seed(seed)
    optimizer_obj = init_optimizer(optimizer_name, optimizer, model_wrapper, model_config, seed)
    optimizer_obj.set_logging_util(LoggingUtil(log_filename))

    start = time.time()
    optimizer_obj.optimize()
    elapsed = time.time() - start

    return elapsed, optimizer_obj.best_config, optimizer_obj.best_value


def get_file_name(path):
    return os.path.splitext(os.path.basename(path))[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Runner")

    parser.add_argument('--datasets', type=str)
    parser.add_argument('--output_directory', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--repeats', type=int)
    parser.add_argument('--budget', type=int, nargs='+', default=None)
    parser.add_argument('--runs_output_folder', type=str)
    parser.add_argument('--logging_folder', type=str)

    args = parser.parse_args()

    optimizer_config = {
        "name": args.name,
        "output_directory": args.output_directory,
    }

    run_experiment(
        datasets=args.datasets,
        optimizers=[optimizer_config],
        repeats=args.repeats,
        checkpoints=args.budget,
        tmp_output_dir=args.runs_output_folder,
        logging_dir=args.logging_folder
    )
