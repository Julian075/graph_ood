class Config:
    def __init__(self, feature_dir: str, class_mapping: str, prompt_template: str, device: str):
        self.feature_dir = feature_dir
        self.class_mapping = class_mapping
        self.prompt_template = prompt_template
        self.device = device
        self.clip_model = "ViT-B/16"