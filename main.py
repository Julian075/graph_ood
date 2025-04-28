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
    parser.add_argument('--synthetic_dir2', type=str, required=False,
                      help='Directory 2 to store synthetic images')
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

def create_config(args, num_classes=None):
    """Create configuration object with consistent parameters across models"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load prompt templates
    if args.prompt_template.endswith('.json'):
        with open(args.prompt_template, 'r') as f:
            prompt_templates = json.load(f)
    else:
        prompt_templates = args.prompt_template
    
    # Base configuration
    config = Config(
        feature_dir=args.feature_dir,
        feature_dir_ood=args.feature_dir_ood,
        class_mapping=args.class_mapping,
        prompt_template=prompt_templates,
        use_synthetic_data=args.use_synthetic_data,
        seed=args.seed,
        device=device,
        num_classes=num_classes
    )
    
    # Common training parameters
    common_params = {
        'learning_rate': 0.001,
        'num_epochs': 100,
        'batch_size': 32,
        'temperature': 0.07,
        'reduction_factor': 0.5,
        'patience': 10
    }
    
    # Model-specific configurations
    config.clip_adapter = common_params.copy()
    config.clip_adapter_graph = common_params.copy()
    config.clip_adapter_graph_simple = common_params.copy()
    config.clip_adapter_ood = common_params.copy()
    
    # Add GNN-specific parameters
    config.clip_adapter_graph.update({
        'gnn_hidden_dim': 512,
        'num_gnn_layers': 2
    })
    
    return config

def main():
    args = parse_args()
    set_global_seed(args.seed)
    
    # Get classes from training directory
    print("\nGetting classes from training directory...")
    if args.input_dir.endswith('VLCS') or args.input_dir.endswith('PACS'):
        folder = os.listdir(args.input_dir)
        train_folder = os.path.join(args.input_dir, folder[0])
    else:
        train_folder = os.path.join(args.input_dir, "train")
    classes = get_classes_from_folder(train_folder, args.class_mapping)
    print(f"Found {len(classes)} classes")
    
    if args.OOD_test:
        classes_ood = get_classes_from_folder(args.input_dir_ood, args.class_mapping)
        print(f"Found {len(classes_ood)} classes in OOD dataset")
    
    # Create configuration after getting classes
    config = create_config(args, num_classes=len(classes))
    
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
                prompt_templates=args.prompt_templates,
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
        if not os.path.isdir(args.feature_dir):
            os.makedirs(args.feature_dir, exist_ok=True)
            
        # Extract features 
        feature_extractor = FeatureExtractor(classes=classes, device=config.device, model_name=config.clip_model, batch_size=32)

        print('extracting features')
        # Process main directory with splits
        features = feature_extractor.process_directory(args.input_dir)
        torch.save(features, os.path.join(args.feature_dir, 'real_data.pt'))

        if args.use_synthetic_data:
            print('extracting features from synthetic data')
            if os.path.isdir(args.synthetic_dir):
                # Process synthetic directory (flat structure)
                synthetic_features = feature_extractor.process_synthetic_directory(args.synthetic_dir)
                torch.save(synthetic_features, os.path.join(args.feature_dir, 'synthetic_features.pt'))
            if os.path.isdir(args.synthetic_dir2):
                # Process second synthetic directory (flat structure)
                synthetic_features2 = feature_extractor.process_synthetic_directory(args.synthetic_dir2)
                torch.save(synthetic_features2, os.path.join(args.feature_dir, 'synthetic_features_diverse.pt'))
    elif args.mode == 'clip_test':
        clip_evaluator = ClipEvaluator(classes=classes, config=config)
        clip_evaluator.evaluate()
    elif args.mode == 'train_clip_adapter':
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
        clip_trainer = CLIPAdapterGraphTrainer(config)
        best_val_acc, checkpoint_path = clip_trainer.train(classes_names=classes)
        
        # Test phase
        clip_adapter_graph = CLIPAdapterGraph(
            reduction_factor=config.clip_adapter['reduction_factor'],
            seed=config.clip_adapter['seed'],
            device=config.device,
            gnn_hidden_dim=config.clip_adapter['gnn_hidden_dim'],
            num_gnn_layers=config.clip_adapter['num_gnn_layers']
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
        clip_trainer = CLIPAdapterGraphSimpleTrainer(config)
        best_val_acc, checkpoint_path = clip_trainer.train(classes_names=classes)
        
        # Test phase
        clip_adapter_graph = CLIPAdapterGraphSimple(
            reduction_factor=config.clip_adapter_graph['reduction_factor'],
            seed=config.clip_adapter_graph['seed'],
            device=config.device
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
