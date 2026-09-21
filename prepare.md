# Project Preparation Guide

This guide outlines the steps required to set up the environment and prepare the dataset for training and evaluating the **Mangrove Segmentation Hybrid Model (TCCFNet)**.

## 1. Environment Setup

The project is built using Python and PyTorch. Follow these steps to set up a clean environment:

### Prerequisites
- Python 3.10 or higher (Python 3.13 recommended)
- CUDA-enabled GPU (optional, but highly recommended for training)

### Setup Steps
1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd antigravitymangrove
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment**:
   - On Linux/macOS:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 2. Dataset Preparation

The training script expects the dataset to be organized in the `dataset_patches` directory (configurable in `config.py`).

### Directory Structure
Ensure your data is organized as follows:
```text
dataset_patches/
├── train/
│   ├── images/  # RGB images (.jpg, .png, .tif)
│   └── masks/   # Binary masks (.jpg, .png, .tif)
└── val/
    ├── images/  # RGB validation images
    └── masks/   # Binary validation masks
```

### Data Requirements
- **Images**: RGB patches (e.g., 224x224 or larger). The model will resize them to the size specified in `config.py`.
- **Masks**: Grayscale images where mangrove pixels are marked with high values (e.g., 255) and non-mangrove pixels are 0. The dataloader automatically converts these to binary labels (0 and 1).
- **Matching Files**: Ensure that each image in `images/` has a corresponding mask in `masks/` with the exact same filename.

## 3. Configuration

Hyperparameters and paths are centralized in `config.py`. Before starting training, you may want to review or modify the following:

- `DATA_DIR`: Path to your dataset patches.
- `IMAGE_SIZE`: Target input size for the model (default: 224).
- `BATCH_SIZE`: Adjust based on your GPU memory (default: 16).
- `LEARNING_RATE`: Initial learning rate for Adam optimizer.
- `NUM_EPOCHS`: Number of training iterations.

## 4. Initializing Training

Once the environment and data are ready, you can start the training process.

### Basic Training
```bash
python train.py
```

### Resuming from Checkpoint
If training was interrupted or you want to continue from the best saved model:
```bash
python train.py --resume
```

### Monitoring
Training logs are printed to the console. If you are running in the background, you can check `training.log`:
```bash
tail -f training.log
```
