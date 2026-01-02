# Oxford-IIIT Pet Classification

### Convolutional Neural Network with Squeeze-and-Excitation Blocks

**Dataset**: Oxford-IIIT Pet
**Task**: Multiclass image classification (37 classes)
**Input resolution**: 128×128
**Architecture**: Custom CNN + SE blocks
**Evaluation split**: stratified train/val, official test set

---

## 1. Introduction

This project addresses a **fine-grained image classification task** involving 37 breeds of cats and dogs from the Oxford-IIIT Pet dataset. The objective is not only to achieve strong classification accuracy, but also to **analyze model behavior**, **training stability**, and **confidence calibration**.

Rather than relying solely on top-1 accuracy, we emphasize:

* robustness across classes (macro metrics),
* calibration quality,
* error structure and interpretability.

---

## 2. Dataset & Preprocessing

### 2.1 Dataset characteristics

The Oxford-IIIT Pet dataset contains:

* 37 classes (12 cats, 25 dogs),
* high intra-class variability (pose, lighting),
* inter-class similarity (related breeds).

Class distributions for train/val and test splits are shown below.

<p align="center">
  <img src="figures/class_dist_trainval.png" width="45%">
  <img src="figures/class_dist_test.png" width="45%">
</p>

The dataset is **approximately balanced**, which justifies the use of macro-averaged metrics.

---

### 2.2 Preprocessing and augmentation

All images are:

* resized to 144×144,
* center-cropped to 128×128,
* normalized using ImageNet statistics.

Training augmentation includes:

* random resized crop,
* horizontal flip,
* rotation (±15°),
* color jitter,
* random erasing (p = 0.25).

Examples of transformed images are shown below.

<p align="center">
  <img src="figures/samples_train_after_aug.png" width="45%">
  <img src="figures/samples_val_after_preprocess.png" width="45%">
</p>

---

## 3. Model Architecture

The model is a **three-stage convolutional network** augmented with **Squeeze-and-Excitation (SE) blocks**.

Each stage consists of:

* Conv → BatchNorm → ReLU → SE
* MaxPooling between stages
* Global Average Pooling before the classifier

Key hyperparameters:

* Channels: 64 → 128 → 256
* Blocks per stage: 2
* SE reduction ratio: 8
* Dropout: 0.2

This architecture balances **capacity** and **regularization**, avoiding overfitting while retaining discriminative power.

---

## 4. Optimization Strategy

### 4.1 Learning rate selection

A learning-rate finder was used to determine a suitable base learning rate.

<p align="center">
  <img src="figures/tb_export/A/train__loss.png" width="45%">
</p>

The loss decreases smoothly up to approximately **2e-3**, after which divergence begins.
We therefore selected **lr = 0.002**.

---

### 4.2 Scheduler and regularization

Training uses:

* AdamW optimizer (weight decay = 0.005),
* linear warmup (5 epochs),
* cosine annealing to a minimum lr of 1e-5.

---

## 5. Experimental Configurations

Two main configurations were evaluated:

| Experiment | Label smoothing | Mixup      |
| ---------- | --------------- | ---------- |
| A          | ❌ disabled      | ✅ enabled  |
| B          | ✅ 0.1           | ❌ disabled |

This design allows isolating the effect of **label smoothing vs mixup**.

---

## 6. Training Dynamics

### 6.1 Loss evolution

<p align="center">
  <img src="figures/tb_export/A/train__loss.png" width="45%">
  <img src="figures/tb_export/B/train__loss.png" width="45%">
</p>

**Observation**:

* Configuration B exhibits **smoother loss curves**.
* Label smoothing stabilizes optimization by preventing overconfident gradients.

---

### 6.2 Validation accuracy and macro-F1

<p align="center">
  <img src="figures/tb_export/A/val__acc1.png" width="45%">
  <img src="figures/tb_export/B/val__acc1.png" width="45%">
</p>

<p align="center">
  <img src="figures/tb_export/A/val__macro_f1.png" width="45%">
  <img src="figures/tb_export/B/val__macro_f1.png" width="45%">
</p>

Both configurations converge to **similar performance**, indicating that:

* label smoothing improves stability,
* but does not significantly alter final accuracy in this setting.

---

## 7. Final Test Results (Configuration B)

| Metric            | Value      |
| ----------------- | ---------- |
| Loss              | **1.5019** |
| Acc@1             | **0.5707** |
| Acc@5             | **0.8735** |
| Macro F1          | **0.5625** |
| Weighted F1       | **0.5626** |
| Balanced accuracy | **0.5706** |
| Macro precision   | **0.5721** |
| Macro recall      | **0.5706** |
| ECE               | **0.0318** |

These results show **consistent behavior across metrics**, with no major class imbalance effects.

---

## 8. Confusion Analysis

### 8.1 Confusion matrix

<p align="center">
  <img src="figures/eval_extra/confusion_matrix_test.png" width="70%">
</p>

The diagonal dominance confirms correct classification for most classes.
Errors are concentrated between **visually similar breeds**.

Top confusion pairs:

* class 27 → 9 (36)
* class 22 → 30 (28)
* class 9 → 27 (23)

---

### 8.2 Per-class accuracy

<p align="center">
  <img src="figures/eval_extra/per_class_accuracy_test.png" width="70%">
</p>

Performance is relatively uniform, validating the macro-F1 score as a reliable indicator.

---

## 9. Calibration Analysis

### 9.1 Confidence distribution

<p align="center">
  <img src="figures/eval_extra/confidence_hist_test.png" width="70%">
</p>

Incorrect predictions tend to have **lower confidence**, indicating that the model is not blindly overconfident.

---

### 9.2 Reliability diagram

<p align="center">
  <img src="figures/reliability_diagram.png" width="60%">
</p>

With **ECE = 0.0318**, the model is **well calibrated**, especially for a CNN trained without temperature scaling.

---

## 10. Error Inspection

### 10.1 Most confident misclassifications

<p align="center">
  <img src="figures/eval_extra/misclassified_grid_test.png" width="80%">
</p>

These errors typically involve:

* extreme lighting,
* partial occlusion,
* ambiguous poses.

This suggests **data ambiguity rather than model failure**.

---

## 11. Discussion

Key findings:

* SE blocks improve feature recalibration without excessive complexity.
* Label smoothing stabilizes training but does not drastically change final performance.
* Mixup improves robustness but slightly increases optimization noise.
* Calibration quality is strong without post-hoc correction.

Limitations:

* Resolution limited to 128×128.
* No pretrained backbone.
* No test-time augmentation.

---

## 12. Conclusion

This project demonstrates a **well-structured experimental pipeline**, emphasizing:

* reproducibility,
* interpretability,
* metric diversity beyond accuracy.

The final model achieves competitive performance while maintaining good calibration and robust behavior across classes.

---

## 13. Reproducibility

All experiments can be reproduced using:

```bash
python -m src.train --config configs/config.yaml --experiment_name B
python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/B_best.ckpt
```

