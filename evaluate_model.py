import sys
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew, kurtosis, entropy

# -------------------- Feature Engineering --------------------
def add_engineered_features(X_raw):
    rcs_hist = np.hstack([X_raw[:, 0:12], X_raw[:, 20:32]])
    range_hist = np.hstack([X_raw[:, 12:20], X_raw[:, 32:40]])

    rcs_mean = rcs_hist.mean(axis=1, keepdims=True)
    rcs_std = rcs_hist.std(axis=1, keepdims=True)
    rcs_skew = skew(rcs_hist, axis=1).reshape(-1, 1)
    rcs_kurt = kurtosis(rcs_hist, axis=1).reshape(-1, 1)
    rcs_entropy = entropy(rcs_hist.T + 1e-9).reshape(-1, 1)

    range_mean = range_hist.mean(axis=1, keepdims=True)
    range_std = range_hist.std(axis=1, keepdims=True)
    range_entropy = entropy(range_hist.T + 1e-9).reshape(-1, 1)

    radar_power = X_raw[:, 68:69]
    rcs_norm = rcs_mean / (radar_power + 1e-6)

    moving_rcs_avg = X_raw[:, 53:54]
    stationary_rcs_avg = X_raw[:, 56:57]
    mov_stat_rcs_ratio = moving_rcs_avg / (stationary_rcs_avg + 1e-6)

    moving_rcs_std = X_raw[:, 54:55]
    stationary_rcs_std = X_raw[:, 57:58]
    mov_stat_std_ratio = moving_rcs_std / (stationary_rcs_std + 1e-6)

    vel_avg = np.abs(X_raw[:, 64:65])
    curve_avg = X_raw[:, 66:67]
    curvature_norm = curve_avg / (vel_avg + 1e-6)

    X_extra = np.hstack([
        rcs_mean, rcs_std, rcs_skew, rcs_kurt, rcs_entropy,
        range_mean, range_std, range_entropy,
        rcs_norm, mov_stat_rcs_ratio, mov_stat_std_ratio,
        curvature_norm
    ])
    return np.hstack([X_raw, X_extra])


# -------------------- Main --------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_model.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    data = pd.read_csv(csv_path, header=None).values
    Y = np.round(data[:, 0])
    X_raw = data[:, 1:].astype(float)
    X = add_engineered_features(X_raw)

    # Load models and scalers
    scaler = joblib.load("artifacts/scaler.pkl")
    encoder_bin = joblib.load("artifacts/encoder_bin.pkl")
    encoder_multi = joblib.load("artifacts/encoder_multi.pkl")
    model_bin = load_model("artifacts/model_bin.keras")
    model_multi = load_model("artifacts/model_multi.keras")

    X_scaled = scaler.transform(X)

    # Binary classification
    y_pred_bin_probs = model_bin.predict(X_scaled)
    y_pred_bin = np.argmax(y_pred_bin_probs, axis=1)

    # Multiclass classification only for non-zero predictions
    nonzero_indices = np.where(y_pred_bin == 1)[0]
    final_pred = np.zeros_like(Y)

    # Save per-class probabilities
    probs_full = np.zeros((len(X), 6))  # For 6 total classes
    label_encoder_all = LabelEncoder()
    label_encoder_all.fit([0.0, 5.0, 14.0, 19.0, 23.0, 29.0])

    probs_full[:, 0] = y_pred_bin_probs[:, 0]  # class 0 prob from binary model

    mapped_indices = label_encoder_all.transform(encoder_multi.inverse_transform(np.arange(5)))
    if nonzero_indices.size > 0:
        X_multi_input = X_scaled[nonzero_indices]
        Y_multi_input = Y[nonzero_indices]

        # Filter for valid target classes
        target_classes = [5.0, 14.0, 19.0, 23.0, 29.0]
        mask_multi = np.isin(Y_multi_input, target_classes)
        X_multi_input = X_multi_input[mask_multi]
        Y_multi_input = Y_multi_input[mask_multi]
        # Predict and map back
        # final_pred = np.zeros_like(Y)
        if X_multi_input.shape[0] > 0:
            multi_probs = model_multi.predict(X_multi_input)
            multi_preds = np.argmax(multi_probs, axis=1)
            multi_preds_labels = encoder_multi.inverse_transform(multi_preds)
            final_pred[nonzero_indices[mask_multi]] = multi_preds_labels
            # Correct assignment
            for i, row_idx in enumerate(nonzero_indices[mask_multi]):
                probs_full[row_idx, mapped_indices] = multi_probs[i]
        else:
            print("No samples for multiclass prediction after filtering.")
    else:
        print("No nonzero indices for multiclass prediction.")
    df_pred = pd.DataFrame({
        "True": Y,
        "Predicted": final_pred
    })
    df_pred.to_csv("predictions_true_vs_pred.csv", index=False)
    df_probs = pd.DataFrame(probs_full, columns=[f"Class_{cls}" for cls in label_encoder_all.classes_])
    df_probs.to_csv("predictions_class_probabilities.csv", index=False)

    print("evaluate_model.py complete. CSV files saved.")
