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
        dtype = data.dtype if isinstance(data, torch.Tensor) else type(data[0]) if data else None
        print(f"{indent}{name}: {type(data).__name__}, shape/length: {shape}, dtype: {dtype}")
        if isinstance(data, list) and data:
            print(f"{indent}  First element type: {type(data[0])}")
            if isinstance(data[0], tuple):
                print(f"{indent}  First element content: {data[0]}")

def compare_feature_files():
    # Load Serengeti features
    print("\nSerengeti Features Structure:")
    print("=" * 50)
    serengeti_path = "./data/features/serengeti/real_data.pt"
    serengeti_synthetic_path = "./data/features/serengeti/synthetic_features.pt"
    
    if os.path.exists(serengeti_path):
        serengeti_data = torch.load(serengeti_path)
        print("Real data:")
        print_structure(serengeti_data, "real_data.pt")
    else:
        print("Serengeti features file not found!")
        
    if os.path.exists(serengeti_synthetic_path):
        serengeti_synthetic = torch.load(serengeti_synthetic_path)
        print("\nSynthetic data:")
        print_structure(serengeti_synthetic, "synthetic_features.pt")
    else:
        print("Serengeti synthetic features file not found!")

    # Load Terra features
    print("\nTerra Features Structure:")
    print("=" * 50)
    terra_path = "./data/features/terra/real_data.pt"
    terra_synthetic_path = "./data/features/terra/synthetic_features.pt"
    
    if os.path.exists(terra_path):
        terra_data = torch.load(terra_path)
        print("Real data:")
        print_structure(terra_data, "real_data.pt")
    else:
        print("Terra features file not found!")
        
    if os.path.exists(terra_synthetic_path):
        terra_synthetic = torch.load(terra_synthetic_path)
        print("\nSynthetic data:")
        print_structure(terra_synthetic, "synthetic_features.pt")
    else:
        print("Terra synthetic features file not found!")

if __name__ == "__main__":
    compare_feature_files() 