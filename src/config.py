import yaml

class Config:
    """Load and store experiment configuration from a YAML file."""
    def __init__(self, path: str = "config.yaml"):
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)

        paths = cfg.get('paths', {})
        self.train_csv       = paths['train_csv']
        self.train_audio_dir = paths['train_audio_dir']
        self.taxonomy_csv    = paths['taxonomy_csv']
        self.output_dir      = paths.get('output_dir', 'outputs')

        audio_cfg = cfg.get('audio', {})
        self.sample_rate = audio_cfg.get('sample_rate', 32000)
        self.n_mels      = audio_cfg.get('n_mels', 224)
        self.n_fft       = audio_cfg.get('n_fft', 1024)
        self.hop_length  = audio_cfg.get('hop_length', 320)
        self.fmin        = audio_cfg.get('fmin', 20)
        self.fmax        = audio_cfg.get('fmax', 16000)

        train_cfg            = cfg.get('training', {})
        self.batch_size      = train_cfg.get('batch_size', 32)
        self.epochs          = train_cfg.get('epochs', 10)
        self.learning_rate   = float(train_cfg.get('learning_rate', 1e-3))
        self.weight_decay    = float(train_cfg.get('weight_decay', 1e-4))
        self.val_fraction    = train_cfg.get('val_fraction', 0.0)
        self.segment_length  = float(train_cfg.get('segment_length', 5.0))  # ← new

        model_cfg            = cfg.get('model', {})
        self.backbone        = model_cfg.get('backbone', 'efficientnet_b0')
        self.num_classes     = model_cfg.get('num_classes', 206)

    def to_dict(self) -> dict:
        return self.__dict__

    def save(self, path: str):
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f)