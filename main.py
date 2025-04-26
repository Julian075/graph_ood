import os
import torch
import argparse
import copy
import json
from src.generation.synthetic_data import generate_synthetic_data
from src.generation.synthetic_data_seg import generate_synthetic_data_seg
from src.utils.utils import get_classes_from_folder, set_global_seed
from src.feature_extraction.feature_extractor import FeatureExtractor
from src.evaluation.clip_test import ClipEvaluator
from src.evaluation.clip_adapter_eval import ClipAdapterEvaluator
from src.models.clip_adapter import CLIPAdapter
from src.training.train_adapter import CLIPAdapterTrainer
from src.models.clip_adapter_graph import CLIPAdapterGraph
from src.training.train_adapter_graph import CLIPAdapterGraphTrainer
from src.evaluation.clip_adapter_graph_eval import ClipAdapterGraphEvaluator
from src.models.clip_adapter_ood import CLIPAdapterOOD
from src.training.train_adapter_ood import CLIPAdapterOODTrainer
from src.evaluation.clip_adapter_ood_eval import ClipAdapterOODEvaluator
from src.models.clip_adapter_graph_simple import CLIPAdapterGraphSimple
from src.training.train_adapter_graph_simple import CLIPAdapterGraphSimpleTrainer
from src.evaluation.clip_adapter_graph_simple_eval import ClipAdapterGraphSimpleEvaluator
from src.utils.hyperparameter_search import RandomSearch, SearchSpace
from src.config.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic data generation pipeline")
    
    # Required arguments
    parser.add_argument('--mode', type=str, required=True,
                      help='Mode to run the script in')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing dataset')
    parser.add_argument('--input_dir_ood', type=str, required=False, default="",
                      help='Directory containing OOD dataset')
    parser.add_argument('--use_synthetic_data', type=str, default='False',
                      choices=['True', 'False'],
                      help='Use synthetic data (True/False)')
    parser.add_argument('--synthetic_dir', type=str, required=False,
                      help='Directory to store synthetic images')
    parser.add_argument('--feature_dir', type=str, required=False,
                      help='Directory to store features')
    parser.add_argument('--feature_dir_ood', type=str, required=False,default="",
                      help='Directory to store OOD features')
    parser.add_argument('--class_mapping', type=str, required=True,
                      help='JSON file mapping folder names to class names')
    
    # Optional arguments
    parser.add_argument('--OOD_test', type=str, default='False',
                      choices=['True', 'False'],
                      help='OOD test (True/False)')
    parser.add_argument('--images_per_class', type=int, default=100,
                      help='Number of synthetic images to generate per class')
    parser.add_argument('--prompt_template', type=str, default="a photo of a {}",
                        help="Single prompt template or path to JSON file with list of templates")
    parser.add_argument('--start_idx', type=int, default=None,
                      help='Starting index for generation')
    parser.add_argument('--end_idx', type=int, default=None,
                      help='Ending index for generation')
    parser.add_argument('--seed', type=int, default=1064200250,
                      help='Random seed for reproducibility')
    parser.add_argument('--hyperparameter_search', type=str, default='False',
                      choices=['True', 'False'],
                      help='Hyperparameter search (True/False)')
    parser.add_argument('--use_attention', type=str, default='False',
                      choices=['True', 'False'],
                      help='Use attention maps for segmentation (True/False)')
    
    
    args = parser.parse_args()
    
    # Convert string booleans to actual booleans
    args.use_synthetic_data = args.use_synthetic_data.lower() == 'true'
    args.OOD_test = args.OOD_test.lower() == 'true'
    args.hyperparameter_search = args.hyperparameter_search.lower() == 'true'
    args.use_attention = args.use_attention.lower() == 'true'
    
    return args

def main():
    args = parse_args()

    set_global_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.prompt_template.endswith('.json'):
        with open(args.prompt_template, 'r') as f:
            prompt_templates = json.load(f)
    else:
        prompt_templates = args.prompt_template
    config = Config(args.feature_dir, args.feature_dir_ood, args.class_mapping, prompt_templates, args.use_synthetic_data, args.seed, device)
    
    # Get classes from training directory
    print("\nGetting classes from training directory...")
    if args.input_dir.endswith('VLCS') or args.input_dir.endswith('PACS'):
        folder=os.listdir(args.input_dir)
        train_folder = os.path.join(args.input_dir, folder[0])
    else:
        train_folder = os.path.join(args.input_dir, "train")
    classes = get_classes_from_folder(train_folder, args.class_mapping)
    print(f"Found {len(classes)} classes")
    if args.OOD_test:
        classes_ood = get_classes_from_folder(args.input_dir_ood, args.class_mapping)
        print(f"Found {len(classes_ood)} classes in OOD dataset")
    
    if args.mode == 'generate':
        # Create output directory if it doesn't exist
        os.makedirs(args.synthetic_dir, exist_ok=True)
        
        # Print configuration
        print("\nConfiguration:")
        print(f"  images_per_class: {args.images_per_class}")
        print(f"  prompt_template: {args.prompt_template}")
        print(f"  use_attention: {args.use_attention}")
        if args.start_idx is not None:
            print(f"  start_idx: {args.start_idx}")
        if args.end_idx is not None:
            print(f"  end_idx: {args.end_idx}")
        
        # Generate synthetic data
        print("\nGenerating synthetic images...")
        
        
        if args.use_attention:
            print("Using attention maps for segmentation")
            generate_synthetic_data_seg(
                output_folder=args.synthetic_dir,
                classes=classes,
                images_per_class=args.images_per_class,
                prompt_templates=prompt_templates,
                seed=args.seed,
                start_idx=args.start_idx,
                end_idx=args.end_idx
            )
        else:
            generate_synthetic_data(
                output_folder=args.synthetic_dir,
                classes=classes,
                images_per_class=args.images_per_class,
                prompt_template=args.prompt_template,
                seed=args.seed,
                start_idx=args.start_idx,
                end_idx=args.end_idx
            )
        print("\nSynthetic data generation completed!")
    elif args.mode == 'extract':
        features={}
        # Extract features 
        feature_extractor = FeatureExtractor(classes=classes,device=config.device, model_name=config.clip_model, batch_size=32)

        print('extracting features')
        for split_folder in os.listdir(args.input_dir):
            split_folder_path = os.path.join(args.input_dir, split_folder)
            if os.path.isdir(split_folder_path):
                features[split_folder] = feature_extractor.process_directory(split_folder_path)
        torch.save(features, os.path.join(args.feature_dir, 'real_data.pt'))

        if args.use_synthetic_data:

            print('extracting features from synthetic data')
            if os.path.isdir(args.synthetic_dir):
                    features_synthetic= feature_extractor.process_directory(args.synthetic_dir)
                    torch.save(features_synthetic, os.path.join(args.feature_dir, 'synthetic_features.pt'))
    elif args.mode == 'clip_test':
        clip_evaluator = ClipEvaluator(classes=classes, config=config)
        clip_evaluator.evaluate()
    elif args.mode == 'train_clip_adapter':
        if args.hyperparameter_search:
            search_space = config.clip_adapter['search_space']
            
            # Convert config search spaces to SearchSpace objects
            search_spaces = []
            for param_name, param_config in search_space['search_spaces'].items():
                search_spaces.append(
                    SearchSpace(
                        name=param_name,
                        type=param_config['type'],
                        range=param_config['range'],
                        log_scale=param_config.get('log_scale', False)
                    )
                )
            
            # Initialize random search
            random_search = RandomSearch(
                search_spaces=search_spaces,
                n_trials=search_space['n_trials'],
                metric_name=search_space['metric_name'],
                maximize=search_space['maximize'],
                seed=config.clip_adapter['seed']
            )
            
            # Define training function for hyperparameter search
            def train_with_params(params):
                # Update config with new parameters
                current_config = copy.deepcopy(config)
                for param_name, param_value in params.items():
                    current_config.clip_adapter[param_name] = param_value
                
                # Initialize model and trainer with current parameters
                clip_adapter = CLIPAdapter(
                    reduction_factor=current_config.clip_adapter['reduction_factor'],
                    device=current_config.device
                )
                clip_trainer = CLIPAdapterTrainer(clip_adapter, current_config)
                
                # Train and return validation accuracy
                val_acc = clip_trainer.train(classes_names=classes)
                return val_acc
            
            # Run hyperparameter search
            print("\nStarting hyperparameter search...")
            best_params = random_search.search(
                train_fn=train_with_params,
                output_dir=os.path.join(config.output_dir, 'clip_adapter', 'hyperparameter_search'),
                verbose=True
            )
            
            # Train final model with best parameters
            print("\nTraining final model with best parameters...")
            for param_name, param_value in best_params.items():
                config.clip_adapter[param_name] = param_value
                
            clip_adapter = CLIPAdapter(config.clip_adapter['reduction_factor'], config.clip_adapter['seed'], config.device)
            clip_trainer = CLIPAdapterTrainer(clip_adapter, config)
            clip_trainer.train(classes_names=classes)
        else:
            clip_trainer = CLIPAdapterTrainer(config)
            best_val_acc, checkpoint_path = clip_trainer.train(classes_names=classes)
        
        # Test phase
        clip_adapter = CLIPAdapter(config.clip_adapter['reduction_factor'], config.clip_adapter['seed'], config.device)
        if args.OOD_test:
            clip_evaluator = ClipAdapterEvaluator(model=clip_adapter, classes=classes_ood, ood_test=True, config=config, checkpoint_path=checkpoint_path)
        else:
            clip_evaluator = ClipAdapterEvaluator(model=clip_adapter, classes=classes, ood_test=False, config=config, checkpoint_path=checkpoint_path)
        
        # Load the best checkpoint before evaluation
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        clip_adapter.load_state_dict(checkpoint['model_state_dict'])
        clip_evaluator.model = clip_adapter
        results = clip_evaluator.evaluate()
        print(f"\nBest validation accuracy achieved: {best_val_acc:.4f}")
    elif args.mode == 'train_clip_adapter_graph':
        if args.hyperparameter_search:
            search_space = config.clip_adapter['search_space']
            
            # Convert config search spaces to SearchSpace objects
            search_spaces = []
            for param_name, param_config in search_space['search_spaces'].items():
                search_spaces.append(
                    SearchSpace(
                        name=param_name,
                        type=param_config['type'],
                        range=param_config['range'],
                        log_scale=param_config.get('log_scale', False)
                    )
                )
            
            # Initialize random search
            random_search = RandomSearch(
                search_spaces=search_spaces,
                n_trials=search_space['n_trials'],
                metric_name=search_space['metric_name'],
                maximize=search_space['maximize'],
                seed=config.clip_adapter['seed']
            )
            
            # Define training function for hyperparameter search
            def train_with_params(params):
                # Update config with new parameters
                current_config = copy.deepcopy(config)
                for param_name, param_value in params.items():
                    current_config.clip_adapter[param_name] = param_value
                
                # Initialize model and trainer with current parameters
                clip_adapter_graph = CLIPAdapterGraph(
                    reduction_factor=current_config.clip_adapter_graph['reduction_factor'],
                    seed=current_config.clip_adapter_graph['seed'],
                    device=current_config.device
                )
                clip_trainer = CLIPAdapterGraphTrainer(current_config)
                
                # Train and return validation accuracy
                val_acc = clip_trainer.train(classes_names=classes)
                return val_acc
            
            # Run hyperparameter search
            print("\nStarting hyperparameter search...")
            best_params = random_search.search(
                train_fn=train_with_params,
                output_dir=os.path.join(config.output_dir, 'clip_adapter_graph', 'hyperparameter_search'),
                verbose=True
            )
            
            # Train final model with best parameters
            print("\nTraining final model with best parameters...")
            for param_name, param_value in best_params.items():
                config.clip_adapter_graph[param_name] = param_value
                
            clip_adapter_graph = CLIPAdapterGraph(
                reduction_factor=config.clip_adapter_graph['reduction_factor'],
                seed=config.clip_adapter_graph['seed'],
                device=config.device
            )
            clip_trainer = CLIPAdapterGraphTrainer(config)
            clip_trainer.train(classes_names=classes)
        else:
            clip_trainer = CLIPAdapterGraphTrainer(config)
            best_val_acc, checkpoint_path = clip_trainer.train(classes_names=classes)
        
        # Test phase
        clip_adapter_graph = CLIPAdapterGraph(
            reduction_factor=config.clip_adapter_graph['reduction_factor'],
            seed=config.clip_adapter_graph['seed'],
            device=config.device,
            gnn_hidden_dim=config.clip_adapter_graph['gnn_hidden_dim'],
            num_gnn_layers=config.clip_adapter_graph['num_gnn_layers']
        )
        if args.OOD_test:
            clip_evaluator = ClipAdapterGraphEvaluator(model=clip_adapter_graph, classes=classes_ood, ood_test=True, config=config, checkpoint_path=checkpoint_path)
        else:
            clip_evaluator = ClipAdapterGraphEvaluator(model=clip_adapter_graph, classes=classes, ood_test=False, config=config, checkpoint_path=checkpoint_path)
        
        # Load the best checkpoint before evaluation
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        clip_adapter_graph.load_state_dict(checkpoint['model_state_dict'])
        clip_evaluator.model = clip_adapter_graph
        results = clip_evaluator.evaluate()
        print(f"\nBest validation accuracy achieved: {best_val_acc:.4f}")
    elif args.mode == 'train_clip_adapter_ood':
        config.num_classes = len(classes)
        if args.hyperparameter_search:
            search_space = config.clip_adapter_ood['search_space']
            
            # Convert config search spaces to SearchSpace objects
            search_spaces = []
            for param_name, param_config in search_space['search_spaces'].items():
                search_spaces.append(
                    SearchSpace(
                        name=param_name,
                        type=param_config['type'],
                        range=param_config['range'],
                        log_scale=param_config.get('log_scale', False)
                    )
                )
            
            # Initialize random search
            random_search = RandomSearch(
                search_spaces=search_spaces,
                n_trials=search_space['n_trials'],
                metric_name=search_space['metric_name'],
                maximize=search_space['maximize'],
                seed=config.clip_adapter_ood['seed']
            )
            
            # Define training function for hyperparameter search
            def train_with_params(params):
                # Update config with new parameters
                current_config = copy.deepcopy(config)
                for param_name, param_value in params.items():
                    current_config.clip_adapter_ood[param_name] = param_value
                
                # Initialize trainer with current parameters
                clip_trainer = CLIPAdapterOODTrainer(current_config)
                
                # Train and return validation accuracy
                val_acc = clip_trainer.train(classes_names=classes)
                return val_acc
            
            # Run hyperparameter search
            print("\nStarting hyperparameter search...")
            best_params = random_search.search(
                train_fn=train_with_params,
                output_dir=os.path.join(config.output_dir, 'clip_adapter_ood', 'hyperparameter_search'),
                verbose=True
            )
            
            # Train final model with best parameters
            print("\nTraining final model with best parameters...")
            for param_name, param_value in best_params.items():
                config.clip_adapter_ood[param_name] = param_value
                
            clip_trainer = CLIPAdapterOODTrainer(config)
            clip_trainer.train(classes_names=classes)
        else:
            # Initialize trainer
            clip_trainer = CLIPAdapterOODTrainer(config)
            best_val_acc, checkpoint_path = clip_trainer.train(classes_names=classes)
        
        # Test phase
        if args.OOD_test:
            clip_evaluator = ClipAdapterOODEvaluator(model=clip_trainer.model, classes=classes_ood, ood_test=True, config=config, checkpoint_path=checkpoint_path)
        else:
            clip_evaluator = ClipAdapterOODEvaluator(model=clip_trainer.model, classes=classes, ood_test=False, config=config, checkpoint_path=checkpoint_path)
        
        # Load the best checkpoint before evaluation
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        clip_trainer.model.load_state_dict(checkpoint['model_state_dict'])
        clip_evaluator.model = clip_trainer.model
        results = clip_evaluator.evaluate()
        print(f"\nBest validation accuracy achieved: {best_val_acc:.4f}")
    elif args.mode == 'train_clip_adapter_graph_simple':
        if args.hyperparameter_search:
            search_space = config.clip_adapter_graph['search_space']
            
            # Convert config search spaces to SearchSpace objects
            search_spaces = []
            for param_name, param_config in search_space['search_spaces'].items():
                search_spaces.append(
                    SearchSpace(
                        name=param_name,
                        type=param_config['type'],
                        range=param_config['range'],
                        log_scale=param_config.get('log_scale', False)
                    )
                )
            
            # Initialize random search
            random_search = RandomSearch(
                search_spaces=search_spaces,
                n_trials=search_space['n_trials'],
                metric_name=search_space['metric_name'],
                maximize=search_space['maximize'],
                seed=config.clip_adapter_graph['seed']
            )
            
            # Define training function for hyperparameter search
            def train_with_params(params):
                # Update config with new parameters
                current_config = copy.deepcopy(config)
                for param_name, param_value in params.items():
                    current_config.clip_adapter_graph[param_name] = param_value
                
                # Initialize trainer with current parameters
                clip_trainer = CLIPAdapterGraphSimpleTrainer(current_config)
                
                # Train and return validation accuracy
                val_acc = clip_trainer.train(classes_names=classes)
                return val_acc
            
            # Run hyperparameter search
            print("\nStarting hyperparameter search...")
            best_params = random_search.search(
                train_fn=train_with_params,
                output_dir=os.path.join(config.output_dir, 'clip_adapter_graph_simple', 'hyperparameter_search'),
                verbose=True
            )
            
            # Train final model with best parameters
            print("\nTraining final model with best parameters...")
            for param_name, param_value in best_params.items():
                config.clip_adapter_graph[param_name] = param_value
            
            clip_trainer = CLIPAdapterGraphSimpleTrainer(config)
            clip_trainer.train(classes_names=classes)
        else:
            clip_trainer = CLIPAdapterGraphSimpleTrainer(config)
            best_val_acc, checkpoint_path = clip_trainer.train(classes_names=classes)
        
        # Test phase
        clip_adapter_graph = CLIPAdapterGraphSimple(
            reduction_factor=config.clip_adapter_graph['reduction_factor'],
            device=config.device,
            gnn_hidden_dim=config.clip_adapter_graph['gnn_hidden_dim'],
            num_gnn_layers=config.clip_adapter_graph['num_gnn_layers'],
            seed=config.clip_adapter_graph['seed']
        )
        
        if args.OOD_test:
            clip_evaluator = ClipAdapterGraphSimpleEvaluator(
                model=clip_adapter_graph,
                classes=classes_ood,
                ood_test=True,
                config=config,
                checkpoint_path=checkpoint_path
            )
        else:
            clip_evaluator = ClipAdapterGraphSimpleEvaluator(
                model=clip_adapter_graph,
                classes=classes,
                ood_test=False,
                config=config,
                checkpoint_path=checkpoint_path
            )
        
        # Load the best checkpoint before evaluation
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        clip_adapter_graph.load_state_dict(checkpoint['model_state_dict'])
        clip_evaluator.model = clip_adapter_graph
        results = clip_evaluator.evaluate()
        print(f"\nBest validation accuracy achieved: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
