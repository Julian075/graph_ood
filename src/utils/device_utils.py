import torch

def get_device() -> str:
    """Get the appropriate device (GPU/CPU)."""
    if torch.cuda.is_available():
        print("Using GPU.")
        return "cuda"
    print("No GPU found. Using CPU.")
    return "cpu" 