import os
import json

def load_class_mapping(mapping_file):
    """Load class mapping from JSON file."""
    try:
        with open(mapping_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load class mapping file: {e}")
        return None

def get_classes_from_folder(train_folder, mapping_file=None):
    """Extract classes from training folder structure with optional mapping."""
    if not os.path.exists(train_folder):
        raise ValueError(f"Directory {train_folder} does not exist")
    
    folder_names = [d for d in os.listdir(train_folder) 
                   if os.path.isdir(os.path.join(train_folder, d))]
    
    if not folder_names:
        raise ValueError(f"No class folders found in {train_folder}")
    
    class_mapping = load_class_mapping(mapping_file) if mapping_file else None
    
    classes = []
    for folder_name in folder_names:
        if class_mapping:
            if folder_name.isdigit() and folder_name in class_mapping:
                classes.append((folder_name, class_mapping[folder_name]))
            elif any(v == folder_name for v in class_mapping.values()):
                classes.append((folder_name, folder_name))
            elif folder_name in class_mapping:
                classes.append((folder_name, class_mapping[folder_name]))
            else:
                classes.append((folder_name, folder_name))
        else:
            classes.append((folder_name, folder_name))
    
    return classes

def set_global_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
