# Antigravity Mangrove Segmentation

A hybrid CNN-Transformer (TCCFNet) implementation for automated mangrove segmentation in satellite imagery. This project combines the spatial feature extraction capabilities of ResNet50 with the long-range dependency modeling of Transformers to achieve precise segmentation masks.

## 🚀 Key Features
- **Hybrid Architecture**: ResNet50 backbone + Transformer encoder + Segmentation decoder.
- **Binary Segmentation**: Optimized for detecting mangroves vs. non-mangrove areas.
- **Robust Training**: Supports resuming from checkpoints and background execution.
- **Interactive UI**: Streamlit application for inference and metric visualization.
- **Performance Metrics**: Comprehensive tracking of IoU, Dice Coefficient, Precision, Recall, and Accuracy.

## 🧮 Mathematical Methodology

### 1. Loss Function
The model uses **Binary Cross Entropy with Logits Loss (BCEWithLogitsLoss)**, which combines a Sigmoid layer and the BCE loss in one single class for better numerical stability.

$$L(x, y) = -\frac{1}{N} \sum_{i=1}^N [y_i \cdot \log(\sigma(x_i)) + (1 - y_i) \cdot \log(1 - \sigma(x_i))]$$

Where:
- $x_i$ is the raw logit output from the model.
- $y_i$ is the ground truth label (0 or 1).
- $\sigma(x_i)$ is the Sigmoid function: $\frac{1}{1 + e^{-x_i}}$.

### 2. Loss Calculation (Train & Val)
The loss reported for each epoch is the **weighted average** of batch losses:

$$\text{Epoch Loss} = \frac{\sum_{j=1}^{\text{batches}} (\text{Batch Loss}_j \times \text{Batch Size}_j)}{\text{Total Samples in Split}}$$

### 3. Evaluation Metrics
We use the following metrics to assess segmentation performance:

- **Intersection over Union (IoU)**:
  $$\text{IoU} = \frac{TP}{TP + FP + FN}$$
- **Dice Coefficient (F1-Score)**:
  $$\text{Dice} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$
- **Precision**:
  $$\text{Precision} = \frac{TP}{TP + FP}$$
- **Recall**:
  $$\text{Recall} = \frac{TP}{TP + FN}$$
- **Accuracy**:
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

*Where $TP$ = True Positives, $TN$ = True Negatives, $FP$ = False Positives, and $FN$ = False Negatives (calculated pixel-wise).*

## 🛠 Getting Started

### Preparation
For detailed instructions on setting up your environment and preparing your dataset, please refer to:
👉 **[prepare.md](prepare.md)**

### Training the Model
To start training with default settings:
```bash
python train.py
```

To resume training from the best saved checkpoint:
```bash
python train.py --resume
```

### Visualization and Inference
Launch the interactive Streamlit dashboard to visualize results and run inference on new images:
```bash
streamlit run app.py
```

## 📊 Project Structure
- `models/`: Hybrid model architecture definitions.
- `utils/`: Dataloader, metrics, and inference utilities.
- `dataset_patches/`: Training and validation data (patches).
- `checkpoints/`: Stored model weights.
- `config.py`: Centralized configuration for hyperparameters and paths.

## 📈 Performance Monitoring
The project generates training logs and visualization plots to track progress. Use `plot_metrics.py` to generate performance graphs from `training.log`.

---
*Developed as part of the Mangrove Monitoring Initiative.*
