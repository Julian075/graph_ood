import torch
import os

def print_structure(data, name="", level=0):
    """Recursively print the structure of nested dictionaries and tensors"""
    indent = "  " * level
    if isinstance(data, dict):
        print(f"{indent}{name}:")
        for key, value in data.items():
            print_structure(value, key, level + 1)
    elif isinstance(data, (torch.Tensor, list)):
        shape = data.shape if isinstance(data, torch.Tensor) else len(data)
        print(f"{indent}{name}: {type(data).__name__}, shape/length: {shape}")

def compare_feature_files():
    # Load Serengeti features
    print("\nSerengeti Features Structure:")
    print("=" * 30)
    serengeti_path = "./data/features/serengeti/real_data.pt"
    if os.path.exists(serengeti_path):
        serengeti_data = torch.load(serengeti_path)
        print_structure(serengeti_data, "real_data.pt")
    else:
        print("Serengeti features file not found!")

    # Load Terra features
    print("\nTerra Features Structure:")
    print("=" * 30)
    terra_path = "./data/features/terra/real_data.pt"
    if os.path.exists(terra_path):
        terra_data = torch.load(terra_path)
        print_structure(terra_data, "real_data.pt")
    else:
        print("Terra features file not found!")

if __name__ == "__main__":
    compare_feature_files() 