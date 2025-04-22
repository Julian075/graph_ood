class Config:
    def __init__(self, feature_dir: str, feature_dir_ood: str,class_mapping: str, prompt_template: str, device: str):
        self.feature_dir = feature_dir
        self.feature_dir_ood = feature_dir_ood
        self.output_dir = "./checkpoints"
        self.class_mapping = class_mapping
        self.prompt_template = prompt_template
        self.device = device
        self.clip_model = "ViT-B/16"
        self.clip_adapter = {
            "reduction_factor": 8,
            "learning_rate": 1e-2,
            "batch_size": 32,
            "temperature": 0.07,
            "num_epochs": 10,
            "patience": 5,
            "seed": 42,
            "random_search": {
                "n_trials": 20,
                "metric_name": "accuracy",
                "maximize": True,
                "search_spaces": {
                    "reduction_factor": {"type": "int", "range": [4, 16, 32]},
                    "learning_rate": {"type": "float", "range": [1e-3, 1e-1]},
                    "batch_size": {"type": "int", "range": [16, 32, 64]},
                    "temperature": {"type": "float", "range": [0.01, 0.1]},
                    "num_epochs": {"type": "int", "range": [5, 10, 20]},
                    "patience": {"type": "int", "range": [3, 5, 10]}
                }
            }
        }