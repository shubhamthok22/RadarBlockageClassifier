import numpy as np
import scipy.stats as st
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, hamming_loss, precision_score, recall_score, f1_score

# -------------------- Create Artifact Directory --------------------
save_dir = os.path.join(os.getcwd(), "artifacts")
os.makedirs(save_dir, exist_ok=True)

# -------------------- Feature Engineering --------------------
def add_engineered_features(X_raw):
    rcs_hist = np.hstack([X_raw[:, 0:12], X_raw[:, 20:32]])
    range_hist = np.hstack([X_raw[:, 12:20], X_raw[:, 32:40]])

    rcs_mean = rcs_hist.mean(axis=1, keepdims=True)
    rcs_std = rcs_hist.std(axis=1, keepdims=True)
    rcs_skew = st.skew(rcs_hist, axis=1).reshape(-1, 1)
    rcs_kurt = st.kurtosis(rcs_hist, axis=1).reshape(-1, 1)
    rcs_entropy = st.entropy(rcs_hist.T + 1e-9).reshape(-1, 1)

    range_mean = range_hist.mean(axis=1, keepdims=True)
    range_std = range_hist.std(axis=1, keepdims=True)
    range_entropy = st.entropy(range_hist.T + 1e-9).reshape(-1, 1)

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
        rcs_norm,
        mov_stat_rcs_ratio, mov_stat_std_ratio,
        curvature_norm
    ])
    return np.hstack([X_raw, X_extra])

# -------------------- Load and Prepare Data --------------------
data = np.load("Complete_Dataset_Updated.npy")
Y = np.round(data[:, 0])
X_raw = data[:, 1:].astype(float)
X = add_engineered_features(X_raw)

Y_bin = (Y != 0).astype(int)
encoder_bin = LabelEncoder()
encoded_Y_bin = encoder_bin.fit_transform(Y_bin)
y_encoded_bin = to_categorical(encoded_Y_bin)

joblib.dump(encoder_bin, os.path.join(save_dir, "encoder_bin.pkl"))

# -------------------- Train-Validation Split --------------------
sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
train_idx, val_idx = next(sss_outer.split(X, encoded_Y_bin))
X_train, X_val = X[train_idx], X[val_idx]
y_train_bin, y_val_bin = y_encoded_bin[train_idx], y_encoded_bin[val_idx]
Y_train = Y[train_idx]
Y_val = Y[val_idx]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))

# -------------------- Binary Classifier --------------------
model_bin = Sequential([
    Dense(256, activation='elu', input_shape=(X_train.shape[1],)),
    BatchNormalization(), Dropout(0.3),
    Dense(128, activation='elu'), BatchNormalization(), Dropout(0.3),
    Dense(64, activation='elu'), Dropout(0.2),
    Dense(2, activation='softmax')
])
model_bin.compile(optimizer=Adam(0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
]

print("\nTraining Binary Classifier...")
history_bin = model_bin.fit(X_train, y_train_bin,
                            validation_data=(X_val, y_val_bin),
                            epochs=50, batch_size=32,
                            callbacks=callbacks, verbose=1)
model_bin.save(os.path.join(save_dir, "model_bin.keras"))
np.save(os.path.join(save_dir, "history_bin.npy"), history_bin.history)

# -------------------- Multiclass Classifier --------------------
class_weight_dict = {
    0: 2.4,  # class 5.0
    1: 1.6,  # class 14.0
    2: 1.5,  # class 19.0
    3: 1.0,  # class 23.0
    4: 1.3   # class 29.0
}

target_classes = [5.0, 14.0, 19.0, 23.0, 29.0]
mask_train = np.isin(Y_train, target_classes)
mask_val = np.isin(Y_val, target_classes)

X_train_nonzero = X_train[mask_train]
Y_train_nonzero = Y_train[mask_train]
X_val_nonzero = X_val[mask_val]
Y_val_nonzero = Y_val[mask_val]

encoder_multi = LabelEncoder()
y_train_multi = to_categorical(encoder_multi.fit_transform(Y_train_nonzero))
y_val_multi = to_categorical(encoder_multi.transform(Y_val_nonzero))
joblib.dump(encoder_multi, os.path.join(save_dir, "encoder_multi.pkl"))

model_multi = Sequential([
    Dense(256, activation='elu', input_shape=(X_train_nonzero.shape[1],)),
    BatchNormalization(), Dropout(0.3),
    Dense(128, activation='elu'), BatchNormalization(), Dropout(0.3),
    Dense(64, activation='elu'), Dropout(0.2),
    Dense(len(np.unique(Y_train_nonzero)), activation='softmax')
])
model_multi.compile(optimizer=Adam(0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

callbacks_multi = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

print("\nTraining Multiclass Classifier...")
history_multi = model_multi.fit(X_train_nonzero, y_train_multi,
                                validation_data=(X_val_nonzero, y_val_multi),
                                epochs=50, batch_size=32,
                                class_weight=class_weight_dict,
                                callbacks=callbacks_multi, verbose=1)
model_multi.save(os.path.join(save_dir, "model_multi.keras"))
np.save(os.path.join(save_dir, "history_multi.npy"), history_multi.history)

print("\nTraining Complete. Models and Artifacts Saved.")
