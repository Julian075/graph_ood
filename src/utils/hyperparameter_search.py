import os
import torch
import numpy as np
import random
import logging
from pathlib import Path
import json
from typing import Callable, Dict, Any, Tuple, Optional, Union, List
import copy
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class SearchSpace:
    """Define a parameter search space"""
    name: str
    type: str  # 'float', 'int', 'categorical'
    range: Union[List[Any], tuple]  # List for categorical, tuple (min, max) for numeric
    log_scale: bool = False  # Whether to sample on log scale for numeric values

class RandomSearch:
    def __init__(
        self,
        search_spaces: List[SearchSpace],
        n_trials: int,
        metric_name: str,
        maximize: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize random search.
        
        Args:
            search_spaces: List of SearchSpace objects defining parameter ranges
            n_trials: Number of random trials to run
            metric_name: Name of the metric to optimize
            maximize: Whether to maximize (True) or minimize (False) the metric
            seed: Random seed for reproducibility
        """
        self.search_spaces = search_spaces
        self.n_trials = n_trials
        self.metric_name = metric_name
        self.maximize = maximize
        self.best_params = None
        self.best_score = float('-inf') if maximize else float('inf')
        self.results = []
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
    
    def _sample_parameter(self, space: SearchSpace) -> Any:
        """Sample a single parameter from its search space"""
        if space.type == 'categorical':
            return random.choice(space.range)
        
        min_val, max_val = space.range
        if space.log_scale:
            log_min = np.log(min_val)
            log_max = np.log(max_val)
            value = np.exp(random.uniform(log_min, log_max))
        else:
            value = random.uniform(min_val, max_val)
            
        if space.type == 'int':
            value = int(round(value))
            
        return value
    
    def _sample_parameters(self) -> Dict[str, Any]:
        """Sample a complete set of parameters"""
        return {
            space.name: self._sample_parameter(space)
            for space in self.search_spaces
        }
    
    def search(
        self,
        train_fn: Callable[[Dict[str, Any]], float],
        output_dir: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Perform random search.
        
        Args:
            train_fn: Function that takes parameters dict and returns metric value
            output_dir: Optional directory to save results
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing best parameters found
        """
        for trial in tqdm(range(self.n_trials), desc="Random Search"):
            # Sample parameters
            params = self._sample_parameters()
            
            if verbose:
                print(f"\nTrial {trial + 1}/{self.n_trials}")
                print("Parameters:", params)
            
            # Train and evaluate
            try:
                score = train_fn(params)
                
                # Update results
                result = {
                    'trial': trial,
                    'parameters': params,
                    self.metric_name: score
                }
                self.results.append(result)
                
                # Update best if better
                is_better = score > self.best_score if self.maximize else score < self.best_score
                if is_better:
                    self.best_score = score
                    self.best_params = copy.deepcopy(params)
                    
                    if verbose:
                        print(f"New best {self.metric_name}: {score:.4f}")
                
                # Save intermediate results
                if output_dir:
                    self._save_results(output_dir)
                    
            except Exception as e:
                print(f"Error in trial {trial + 1}: {str(e)}")
                continue
        
        if verbose:
            print("\nSearch completed!")
            print(f"Best {self.metric_name}: {self.best_score:.4f}")
            print("Best parameters:", self.best_params)
        
        return self.best_params
    
    def _save_results(self, output_dir: str):
        """Save search results to JSON file"""
        os.makedirs(output_dir, exist_ok=True)
        
        results_dict = {
            'best_parameters': self.best_params,
            f'best_{self.metric_name}': self.best_score,
            'all_trials': self.results
        }
        
        output_file = os.path.join(output_dir, 'random_search_results.json')
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)