# MLOWLS Training - BirdCLEF 2025

A robust training pipeline for BirdCLEF 2025 with SOLID architecture, featuring multiple training strategies, comprehensive metrics, and professional experiment tracking.

## 📂 Project Structure

```text
MLOWLS_Training/
├── 📂 data/                     # Raw metadata & audio files
│   └── 📂 birdclef-2025/
│       ├── 📜 train.csv
│       ├── 📜 taxonomy.csv
│       └── 🎵 train_audio/
├── 🧪 mlruns/                   # MLflow tracking data
├── 📓 notebooks/                # EDA and prototyping
│   └── 📊 eda.ipynb
├── 📤 outputs/                  # Model checkpoints & artifacts
├── 🛠️ config.yaml              # Configuration file
├── 📦 pyproject.toml            # Project dependencies & metadata
├── 🔧 .gitignore               # Git ignore patterns
├── 📖 README.md                # This file
└── 🧩 src/                     # Source code package (SOLID Architecture)
    ├── 🏛️ interfaces/          # Abstract contracts (Dependency Inversion)
    │   ├── 📋 logger.py         # Logger interface
    │   ├── 📊 metric_calculator.py # Metric calculation interface
    │   ├── 💾 model_saver.py    # Model saving interface
    │   └── 🎯 training_strategy.py # Training strategy interface
    ├── 📈 metrics/              # Metric implementations
    │   ├── 🎯 accuracy.py       # Accuracy calculator (with top-k support)
    │   ├── 📊 auc.py            # AUC calculator (handles missing classes)
    │   └── 🔢 aggregator.py     # Metrics aggregator
    ├── 🏋️ training/             # Training orchestration
    │   ├── 🎲 strategies.py     # Training strategies (Standard, Mixup, CutMix)
    │   ├── 📅 epoch_trainer.py  # Single epoch training logic
    │   └── 🎯 trainer.py        # Main training orchestrator
    ├── 💾 persistence/          # Model saving implementations
    │   └── 🗄️ model_saver.py    # PyTorch model saver
    ├── ⚙️ config.py             # Configuration loader
    ├── 📂 dataset.py            # BirdClefDataset with segment generation
    ├── 🧰 utils.py              # Audio transforms & augmentations
    ├── 📊 mlflow_logger.py      # MLflow experiment tracking
    ├── 🏭 trainer_factory.py    # Dependency injection factory
    └── 🚂 train.py              # Main training script
```

## ✨ Features

### 🏗️ SOLID Architecture
- **🔒 Single Responsibility**: Each class has one clear purpose
- **📖 Open/Closed**: Easy to extend with new strategies and metrics
- **🔄 Liskov Substitution**: Strategies and metrics are interchangeable
- **🏛️ Interface Segregation**: Clean, focused interfaces
- **⬇️ Dependency Inversion**: High-level modules don't depend on low-level details

### 🎯 Training Strategies
- **🎪 Standard Training**: Classic supervised learning
- **🎭 Mixup Training**: Data augmentation mixing samples and labels
- **✂️ CutMix Training**: Spatial augmentation for spectrograms
- **🔧 Extensible**: Easy to add new training strategies

### 📊 Comprehensive Metrics
- **🎯 Accuracy**: Standard and top-k accuracy
- **📈 AUC-ROC**: Multi-class with proper missing class handling
- **🔢 Aggregated**: Unified metrics calculation and reporting
- **📋 Configurable**: Easy to add custom metrics

### 🎵 Audio Processing
- **🔄 Multi-Segment Training**: Extract multiple overlapping segments per audio file
- **🎛️ SpecAugment**: Frequency and time masking for robustness
- **⚙️ Configurable**: Flexible audio preprocessing parameters

### 📈 Experiment Tracking
- **🔬 MLflow Integration**: Professional experiment tracking
- **📊 Rich Progress Bars**: Beautiful real-time training progress
- **💾 Model Artifacts**: Automatic model saving with metadata
- **📋 Configuration Logging**: Full reproducibility

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10–3.12
- NVIDIA GPU + CUDA (recommended)
- [Kaggle API](https://github.com/Kaggle/kaggle-api) credentials (`~/.kaggle/kaggle.json`)

### 2. Setup Environment

**Option A: Using pip**

```bash
git clone https://github.com/christian-pala/MLOWLS_Training.git
cd MLOWLS_Training

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install project in development mode
pip install -e ".[dev,viz,notebook]"
```

**Option B: Using conda**

```bash
git clone https://github.com/christian-pala/MLOWLS_Training.git
cd MLOWLS_Training

# Create conda environment
conda create -n mlowls python=3.11 -y
conda activate mlowls

# Install PyTorch with CUDA support (adjust CUDA version as needed)
conda install pytorch torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install remaining dependencies
pip install -e ".[dev,viz,notebook]"
```

### 3. Download Data

```bash
# Create data directory
mkdir -p data && cd data

# Download BirdCLEF 2025 dataset
kaggle competitions download -c birdclef-2025 --unzip
cd ..
```

### 4. Configure Training

Edit `config.yaml` to customize:

```yaml
# Training strategy configuration
training:
  batch_size: 32
  epochs: 10
  learning_rate: 1e-3
  weight_decay: 1e-4

  # Choose training strategy
  training_strategy: "standard"    # "standard", "mixup", or "cutmix"
  mixup_alpha: 0.4                # For mixup strategy
  cutmix_alpha: 1.0               # For cutmix strategy

  # Early stopping
  early_stopping_patience: 10

# Metrics configuration
training:
  accuracy_top_k: 1               # Top-k accuracy (1 = standard accuracy)
  auc_average: "macro"            # AUC averaging strategy

# Audio processing
audio:
  segment_length: 30.0            # Segment length in seconds
  overlap: 0.5                    # 50% overlap between segments
  max_segments_per_file: 5        # Max segments per audio file

# Model configuration
model:
  backbone: efficientnet_b0
  num_classes: 206
  dropout: 0.3

# Experiment tracking
experiment:
  name: birdclef25_robust_train
  run_name: null                  # Auto-generated if null
```

### 5. Train Model

```bash
# Using the installed CLI command
mlowls-train --config config.yaml

# Or using module syntax
python -m src.train --config config.yaml

# To run on specific GPU
CUDA_VISIBLE_DEVICES=0 mlowls-train --config config.yaml
```

### 6. Monitor Experiments

```bash
# Start MLflow UI
mlflow ui

# Open browser to http://localhost:5000
```

## 🎯 Training Strategies

The SOLID architecture makes it easy to switch between different training approaches:

### 🎪 Standard Training
```yaml
training:
  training_strategy: "standard"
```
Classic supervised learning without augmentation.

### 🎭 Mixup Training
```yaml
training:
  training_strategy: "mixup"
  mixup_alpha: 0.4              # Controls mixing strength
```
Combines pairs of examples and their labels during training for better generalization.

### ✂️ CutMix Training
```yaml
training:
  training_strategy: "cutmix"
  cutmix_alpha: 1.0             # Controls cut region size
```
Spatial augmentation that cuts and pastes regions between spectrograms.

## 📊 Metrics System

The modular metrics system provides comprehensive evaluation:

### 🎯 Accuracy Metrics
```yaml
training:
  accuracy_top_k: 1             # Standard accuracy
  accuracy_top_k: 5             # Top-5 accuracy for harder evaluation
```

### 📈 AUC Metrics
```yaml
training:
  auc_average: "macro"          # Macro-averaged AUC
  auc_average: "micro"          # Micro-averaged AUC
  auc_average: "weighted"       # Weighted AUC
```

## 🔧 Architecture Benefits

### 🧪 Easy Testing
```python
# Test individual components
from src.metrics.accuracy import AccuracyCalculator
from src.training.strategies import MixupTrainingStrategy

# Each component can be tested in isolation
accuracy_calc = AccuracyCalculator(top_k=5)
mixup_strategy = MixupTrainingStrategy(alpha=0.4)
```

### 🔄 Easy Extension
```python
# Add a new training strategy
class SpecAugmentStrategy(TrainingStrategy):
    def train_step(self, model, batch, optimizer, criterion, device):
        # Your implementation here
        pass

# Add a new metric
class F1Calculator(MetricCalculator):
    def calculate(self, predictions, labels):
        # Your F1 implementation here
        pass
```

### ⚙️ Flexible Configuration
```python
# Create trainer with factory pattern
from src.trainer_factory import TrainerFactory

trainer = TrainerFactory.create_trainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    dataloaders=dataloaders,
    device=device,
    config=config,
    logger=logger
)
```

## 🧪 Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Format code
black src/
isort src/

# Type checking
mypy src/

# Run tests (when available)
pytest
```

## 📈 Performance Tips

1. **🎯 Strategy Selection**:
   - Use `"standard"` for baseline
   - Use `"mixup"` for better generalization
   - Use `"cutmix"` for spatial robustness

2. **💾 GPU Memory**:
   - Reduce `batch_size` if you encounter OOM errors
   - Use smaller backbones (efficientnet_b0 vs b3)

3. **⚡ Training Speed**:
   - Reduce `max_segments_per_file` for faster data loading
   - Use `accuracy_top_k: 1` for faster metric calculation

4. **📊 Validation**:
   - Increase `val_fraction` to 0.2 for more reliable validation metrics
   - Use `early_stopping_patience` to prevent overfitting

## 🏗️ Extending the System

### Adding a New Training Strategy

1. **Create strategy class**:
```python
# src/training/strategies.py
class YourStrategy(TrainingStrategy):
    def train_step(self, model, batch, optimizer, criterion, device):
        # Your implementation
        pass
```

2. **Update factory**:
```python
# src/trainer_factory.py
elif strategy_name == 'your_strategy':
    return YourStrategy(**kwargs)
```

3. **Configure in YAML**:
```yaml
training:
  training_strategy: "your_strategy"
```

### Adding a New Metric

1. **Create metric class**:
```python
# src/metrics/your_metric.py
class YourMetric(MetricCalculator):
    def calculate(self, predictions, labels):
        # Your implementation
        return metric_value
```

2. **Update factory**:
```python
# src/trainer_factory.py
aggregator.add_metric(YourMetric())
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is released under the MIT License. See `LICENSE` file for details.

## 🙏 Acknowledgments

- Built for the Machine Learning in Data Operations 2025 course at SUPSI / ZHAW
- Uses the BirdCLEF 2025 dataset from Kaggle
- Implements SOLID principles for maintainable and extensible code
