import os
import torch
import numpy as np
import random
import logging
from pathlib import Path
import json
from typing import Callable, Dict, Any, Tuple, Optional, Union

def sample_hyperparameters(search_space: Dict[str, Tuple[Any, str]], seed: Optional[int] = None) -> Dict[str, Any]:
    """Sample hyperparameters from the search space
    
    Args:
        search_space: Dictionary mapping parameter names to their search space definition
                     Format: {param_name: (values, scale)} where scale can be:
                     - 'log': logarithmic scale between min and max values
                     - 'linear': linear scale between min and max values
                     - 'choice': discrete choice from list of values
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with sampled parameters
    """
    if seed is not None:
        np.random.seed(seed)
        
    params = {}
    for name, (space_def, scale) in search_space.items():
        if scale == 'choice':
            value = np.random.choice(space_def)
            # Convert numpy types to Python native types
            if isinstance(value, (np.integer, np.floating)):
                value = value.item()
            params[name] = value
        elif scale == 'log':
            min_val, max_val = space_def
            value = np.exp(
                np.random.uniform(np.log(min_val), np.log(max_val))
            )
            params[name] = float(value)  # Convert to Python float
        else:  # linear
            min_val, max_val = space_def
            value = np.random.uniform(min_val, max_val)
            params[name] = float(value)  # Convert to Python float
    
    return params

def setup_random_seeds(seed: int) -> None:
    """Setup all random seeds for reproducibility
    
    Args:
        seed: Seed value to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def random_search(
    train_fn: Callable[[Dict[str, Any]], Tuple[float, str]],
    search_space: Dict[str, Tuple[Any, str]],
    n_trials: int = 20,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
    maximize: bool = True
) -> Tuple[Dict[str, Any], float, str]:
    """Generic random search for hyperparameter optimization
    
    Args:
        train_fn: Training function that takes hyperparameters dict and returns
                 (metric_value, model_path). The metric_value will be maximized/minimized
        search_space: Dictionary defining the search space for each parameter
        n_trials: Number of trials to run
        seed: Random seed for reproducibility
        output_dir: Directory to save search results. If None, results won't be saved
        maximize: Whether to maximize or minimize the metric
        
    Returns:
        Tuple containing:
        - Dictionary with best hyperparameters
        - Best metric value achieved
        - Path to the best model
    """
    # Setup reproducibility if seed provided
    if seed is not None:
        setup_random_seeds(seed)
    
    # Setup logging and results directory if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(output_dir, "search.log"),
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
    
    best_metric = float('-inf') if maximize else float('inf')
    best_config = None
    best_model_path = None
    all_results = []
    
    for trial in range(n_trials):
        # Sample hyperparameters
        trial_seed = seed * 100 + trial if seed is not None else None
        hp = sample_hyperparameters(search_space, seed=trial_seed)
        
        message = f"\nTrial {trial + 1}/{n_trials}\nParameters: {hp}"
        if output_dir:
            logging.info(message)
        print(message)
        
        # Train and evaluate with current hyperparameters
        metric_value, model_path = train_fn(hp)
        
        # Save trial results
        trial_results = {
            'trial': trial,
            'hyperparameters': hp,
            'metric_value': metric_value,
            'model_path': model_path
        }
        all_results.append(trial_results)
        
        # Update best if necessary
        is_better = metric_value > best_metric if maximize else metric_value < best_metric
        if is_better:
            best_metric = metric_value
            best_config = hp.copy()
            best_model_path = model_path
            message = f"New best {'maximum' if maximize else 'minimum'}: {best_metric:.4f}"
            if output_dir:
                logging.info(message)
            print(message)
        
        # Save all results if output directory provided
        if output_dir:
            with open(os.path.join(output_dir, "search_results.json"), 'w') as f:
                json.dump(all_results, f, indent=2)
    
    return best_config, best_metric, best_model_path 