# analysis/frontier_explosion_analysis.py
import pandas as pd
import numpy as np
from pathlib import Path
import json

def load_optimizer_results(results_dir, optimizer_name, budget):
    """
    Load all run JSON files for a specific optimizer, dataset, and budget
    """
    results = []
    result_path = Path(results_dir) / f"results_{optimizer_name}" / optimizer_name
    
    for csv_file in result_path.glob(f"*_{budget}.csv"):
        dataset_name = csv_file.stem.replace(f'_{budget}', '')
        # Each run JSON file in metadata folder: dataset_0.json, dataset_1.json, ...
        metadata_dir = csv_file.parent / "metadata"
        run_files = list(metadata_dir.glob(f"{dataset_name}_*.json"))
        print(run_files)
        run_results = []
        for run_file in run_files:
            if run_file.exists():
                with open(run_file) as f:
                    metadata = json.load(f)
                    run_results.append(metadata)
        
        if run_results:
            results.append({
                'dataset': dataset_name,
                'budget': budget,
                'runs': run_results  # store all runs for averaging
            })
    
    return results

def calculate_frontier_stats(results_dir, dominance_methods, budgets):
    """Calculate frontier explosion statistics averaged across runs"""
    
    stats = []
    
    for method in dominance_methods:
        for budget in budgets:
            method_results = load_optimizer_results(results_dir, method, budget)
            
            for result in method_results:
                run_frontier_percentages = []
                n_objs = None
                
                for run_metadata in result['runs']:
                    if 'frontier_history' in run_metadata and run_metadata['frontier_history']:
                        final_frontier = run_metadata['frontier_history'][-1]
                        run_frontier_percentages.append(final_frontier['frontier_percentage'])
                        # Determine number of objectives from the first evaluation of first run
                        if n_objs is None and 'evaluation_history' in run_metadata and run_metadata['evaluation_history']:
                            n_objs = len(run_metadata['evaluation_history'][0]['objectives'])
                
                if run_frontier_percentages:
                    avg_frontier_percentage = np.mean(run_frontier_percentages)
                    stats.append({
                        'method': method,
                        'budget': budget,
                        'dataset': result['dataset'],
                        'n_objectives': n_objs if n_objs else 3,
                        'frontier_percentage': avg_frontier_percentage
                    })
    
    df = pd.DataFrame(stats)
    
    # Stratify by objective count
    df['obj_category'] = pd.cut(df['n_objectives'], 
                                  bins=[0, 2, 3, 4, 100],
                                  labels=['2', '3', '4', '5+'])
    
    # Aggregate across datasets for each method
    summary = df.groupby(['obj_category', 'method']).agg({
        'frontier_percentage': 'mean',
        'dataset': 'count'
    }).reset_index()
    
    summary.columns = ['Objectives', 'Method', 'Frontier %', 'Dataset Count']
    
    # Pivot for table
    table = summary.pivot(index='Objectives', columns='Method', values='Frontier %')
    
    # Add dataset counts
    counts = summary.groupby('Objectives')['Dataset Count'].first()
    table['Dataset Count'] = counts
    
    return table

# Usage
if __name__ == "__main__":
    results_dir = "../results"
    dominance_methods = ['NSGA2', 'SPEA2', 'MOTPE']
    budgets = [50, 40, 20]
    
    table1 = calculate_frontier_stats(results_dir, dominance_methods, budgets)
    print("\n=== TABLE 1: Frontier Explosion Analysis ===")
    print(table1.round(1))
    table1.to_csv("table1_frontier_explosion.csv")
