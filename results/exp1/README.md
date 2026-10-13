# Experiment 1: CLAHE Contrast Enhancement & Geometric Augmentation

## Experimental Findings
- Sensitivity jumped from 82.73% to 93.31% (+10.58% gain in lesion recall).
- Validation loss dropped from 0.5102 to 0.4528.
- Demonstrates a classic clinical tradeoff: CLAHE enhances low-contrast parenchymal edges, reducing false negatives at the cost of slight boundary over-segmentation.

Artifacts:
- exp1_loss_curves.png
- exp1_confusion_matrix.png
- baseline_vs_exp1_comparison.png
- exp1_training_log.json
