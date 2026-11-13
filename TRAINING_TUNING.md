# Training tuning record

## Previous training result

The most recent committed evaluation artifact was generated from the confusion
matrix in `plot_performance.py`:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.5710 |
| Precision | 0.5192 |
| Recall | 0.8706 |
| IoU | 0.4820 |
| Dice / F1 | 0.6505 |

The result has high recall but low precision. The model is therefore detecting
most mangrove pixels while producing too many false positives (`FP=6,251,050`
versus `FN=1,003,067`). The checked-in `training.log` and
`training_retrain.log` stop during training and do not contain a complete,
validated retraining result.

## Tuning changes

The Colab notebook should run the following candidate after the baseline:

| Setting | Previous value | Candidate |
| --- | ---: | ---: |
| Batch size | 8 | 16 |
| Decoder/transformer learning rate | 1e-4 | 2e-4 |
| Backbone learning rate | 5e-5 | 2e-5 |
| Epochs | 50 | 60 |
| Warmup epochs | 3 | 5 |
| Backbone freeze epochs | 5 | 3 |
| Focal/Dice weights | 0.5 / 0.5 | 0.3 / 0.7 |
| Validation threshold | 0.65 | sweep 0.50–0.90 |

Rationale:

- A higher Dice weight directly emphasizes region overlap and should reduce
  fragmented masks.
- A lower backbone learning rate protects ImageNet features while allowing the
  decoder to adapt faster.
- A shorter freeze and longer warmup give the transformer/decoder a stable
  initialization before full fine-tuning.
- The threshold must be selected on the validation set rather than assumed.
  Because the previous run was false-positive heavy, thresholds above 0.65 are
  plausible candidates, but the selection criterion is maximum validation IoU.

The training script now accepts these values as command-line overrides and
saves the selected validation threshold in the checkpoint metadata. This keeps
the experiment reproducible without changing the default production config.

## Google Colab procedure

1. Open `train_colab.ipynb` in Colab, enable a T4 GPU, clone the repository,
   and mount the dataset as described in the notebook.
2. Run the baseline command and save its log:

   ```bash
   !python train.py --save-path checkpoints/baseline_model.pth 2>&1 | tee training_baseline.log
   ```

3. Run the candidate configuration:

   ```bash
   !python train.py \
     --batch-size 16 \
     --lr 2e-4 \
     --backbone-lr 2e-5 \
     --epochs 60 \
     --warmup-epochs 5 \
     --freeze-epochs 3 \
     --bce-weight 0.3 \
     --dice-weight 0.7 \
     --threshold 0.75 \
     --save-path checkpoints/tuned_model.pth \
     2>&1 | tee training_tuned.log
   ```

4. Run the notebook's validation threshold sweep on the tuned checkpoint. The
   sweep evaluates thresholds from 0.50 through 0.90 and selects the threshold
   with the highest validation IoU. Record the resulting threshold, epoch,
   IoU, Dice, precision, recall, and confusion matrix below.
5. Copy the tuned checkpoint, log, curves, and this updated record to Drive.

## Retraining outcome

**Status: pending Google Colab execution.** This repository does not contain
the dataset and no active Colab runtime is available from the local CLI, so a
new measured metric must not be fabricated here.

| Run | Best epoch | Threshold | IoU | Dice | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous artifact | 2 | 0.65 | 0.4820 | 0.6505 | 0.5192 | 0.8706 |
| Tuned Colab run | — | — | — | — | — | — |
