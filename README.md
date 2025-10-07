## RadarBlockageClassifier

This project provides a deep learning solution for classifying radar data into six classes (0, 5, 14, 19, 23, 29) using a two-stage neural network approach. It is designed for fast and accurate inference on new radar datasets.

### Project Overview
- **Stage 1:** Binary classification (class 0 vs. non-zero classes).
- **Stage 2:** Multiclass classification (among non-zero classes: 5, 14, 19, 23, 29).
- **Feature Engineering:** Custom features are extracted from raw radar data for improved model performance.

---

## Installation
Install all dependencies using:
```sh
pip install -r requirements.txt
```

---

## Usage
Run the following command to generate predictions on your dataset:
```sh
python evaluate_model.py your_training_csv_path.csv
```

**Outputs:**
- `predictions_true_vs_pred.csv`: Contains true vs. predicted class labels for each sample.
- `predictions_class_probabilities.csv`: Contains per-class probability scores for each sample.

---

## File Descriptions
- `train_model.py`: Script used to train the models (for reference only; retraining not required).
- `evaluate_model.py`: Main script for inference and prediction.
- `artifacts/`: Folder containing pretrained models, encoders, scaler, and training histories.
- `requirements.txt`: List of required Python packages.
- `predictions_true_vs_pred.csv`, `predictions_class_probabilities.csv`: Output files generated after running inference.
