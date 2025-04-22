import os
import torch
import argparse
from src.generation.synthetic_data import generate_synthetic_data
from src.utils.utils import get_classes_from_folder
from src.feature_extraction.feature_extractor import FeatureExtractor
from src.evaluation.clip_test import ClipEvaluator
from src.evaluation.clip_adapter_eval import ClipAdapterEvaluator
from src.models.clip_adapter import CLIPAdapter
from src.training.train_adapter import CLIPAdapterTrainer
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
    parser.add_argument('--use_synthetic_data', type=bool,default=False, required=False,
                      help='Use synthetic data')
    parser.add_argument('--synthetic_dir', type=str, required=False,
                      help='Directory to store synthetic images')
    parser.add_argument('--feature_dir', type=str, required=False,
                      help='Directory to store features')
    parser.add_argument('--feature_dir_ood', type=str, required=False,default="",
                      help='Directory to store OOD features')
    parser.add_argument('--class_mapping', type=str, required=True,
                      help='JSON file mapping folder names to class names')
    
    # Optional arguments
    parser.add_argument('--OOD_test', type=bool, default=False,
                      help='OOD test')
    parser.add_argument('--images_per_class', type=int, default=100,
                      help='Number of synthetic images to generate per class')
    parser.add_argument('--prompt_template', type=str, default="a photo of a {}",
                      help='Template for generating prompts')
    parser.add_argument('--start_idx', type=int, default=None,
                      help='Starting index for generation')
    parser.add_argument('--end_idx', type=int, default=None,
                      help='Ending index for generation')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--hyperparameter_search', type=bool, default=False,
                      help='Hyperparameter search')
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = Config(args.feature_dir, args.feature_dir_ood, args.class_mapping, args.prompt_template, device)
    # Get classes from training directory
    print("\nGetting classes from training directory...")
    train_folder = os.path.join(args.input_dir, "train")
    classes = get_classes_from_folder(train_folder, args.class_mapping)
    print(f"Found {len(classes)} classes")
    if args.OOD_test:
        classes_ood = get_classes_from_folder(args.input_dir_ood, args.class_mapping)
        print(f"Found {len(classes_ood)} classes")
    
    if args.mode == 'generate':
        # Create output directory if it doesn't exist
        os.makedirs(args.synthetic_dir, exist_ok=True)
        
        # Print configuration
        print("\nConfiguration:")
        print(f"  images_per_class: {args.images_per_class}")
        print(f"  prompt_template: {args.prompt_template}")
        if args.start_idx is not None:
            print(f"  start_idx: {args.start_idx}")
        if args.end_idx is not None:
            print(f"  end_idx: {args.end_idx}")
        
        # Generate synthetic data
        print("\nGenerating synthetic images...")
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
            a=0 #TO_DO: hyperparameter search
        else:
            clip_adapter = CLIPAdapter(config.clip_adapter['reduction_factor'], config.device)
            clip_trainer = CLIPAdapterTrainer(clip_adapter, config)
            clip_trainer.train(classes_names=classes)
            
            #test
            if args.OOD_test:
                clip_evaluator = ClipAdapterEvaluator(classes=classes_ood, ood_test=True, config=config)
                clip_evaluator.evaluate()
            else:
                clip_evaluator = ClipAdapterEvaluator(classes=classes, ood_test=False, config=config)
                clip_evaluator.evaluate()


if __name__ == "__main__":
    main()
