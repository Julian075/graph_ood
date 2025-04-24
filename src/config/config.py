class Config:
    def __init__(self, feature_dir: str, feature_dir_ood: str,class_mapping: str, prompt_template: str, use_synthetic_data: bool, seed: int, device: str):
        self.feature_dir = feature_dir
        self.feature_dir_ood = feature_dir_ood
        self.output_dir = "./checkpoints"
        self.class_mapping = class_mapping
        self.prompt_template = prompt_template
        self.use_synthetic_data = use_synthetic_data
        self.device = device
        self.clip_model = "ViT-B/16"
        self.clip_adapter = {
            "reduction_factor": 2,
            "learning_rate": 1e-2,
            "batch_size": 128,
            "temperature": 0.1,
            "num_epochs": 107,
            "patience": 5,
            "seed": seed,
            "search_space": {
                "n_trials": 20,
                "metric_name": "accuracy",
                "maximize": True,
                "search_spaces": {
                    "reduction_factor": {"type": "int", "range": [2,4, 16, 32]},
                    "learning_rate": {"type": "float", "range": [1e-3, 1e-1]},
                    "batch_size": {"type": "int", "range": [16, 32, 64]},
                    "temperature": {"type": "float", "range": [0.01, 0.1]},
                    "num_epochs": {"type": "int", "range": [5, 10, 20]},
                    "patience": {"type": "int", "range": [3, 5, 10]}
                }
            }
        }
        self.clip_adapter_graph = {
            "patience": 5,
            "reduction_factor": 2,
            "learning_rate": 1e-2,
            "batch_size": 128,
            "temperature": 0.1,
            "num_epochs": 107,
            "gnn_hidden_dim": 256,
            "num_gnn_layers": 2,
            "seed": seed,
            "search_space": {
                "n_trials": 20,
                "metric_name": "accuracy",
                "maximize": True,
                "search_spaces": {
                    "reduction_factor": {"type": "int", "range": [2,4, 16, 32]},
                    "learning_rate": {"type": "float", "range": [1e-3, 1e-1]},
                    "batch_size": {"type": "int", "range": [16, 32, 64]},
                    "temperature": {"type": "float", "range": [0.01, 0.1]},
                    "num_epochs": {"type": "int", "range": [5, 10, 20]},
                    "gnn_hidden_dim": {"type": "int", "range": [128, 256, 512]},
                    "num_gnn_layers": {"type": "int", "range": [1, 2, 3]}
                }
            }
        }