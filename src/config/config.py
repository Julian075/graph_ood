class Config:
    def __init__(self, feature_dir: str, class_mapping: str, prompt_template: str, device: str):
        self.feature_dir = feature_dir
        self.output_dir = "./checkpoints"
        self.class_mapping = class_mapping
        self.prompt_template = prompt_template
        self.device = device
        self.clip_model = "ViT-B/16"
        self.clip_adapter = {
            "reduction_factor": 16,
            "learning_rate": 0.0001,
            "batch_size": 32,
            "temperature": 1.0,
            "num_epochs": 10,
            "patience": 5,
            "seed": 42
        }