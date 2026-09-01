"""
LSTM Vehicle Dynamics Model
============================
Replaces XGBoost Vehicle Dynamics model.
Uses 150 timesteps (15 seconds) of driving history to predict
velocity, acceleration, and motor torque at the next step.

Why LSTM over XGBoost:
  XGBoost sees only 1 step back (t-1).
  LSTM sees 150 steps — capturing vehicle momentum,
  sustained acceleration/braking cycles, and road gradient
  trends that play out over seconds, not milliseconds.

Sequence design:
  Length  : 150 steps = 15 seconds at 10Hz
  Stride  : 3  (one sequence every 3 steps — 3x memory reduction)
  Features: 6 per step (all temporal signals)

Sequence input features (what each of the 150 steps contains):
  Throttle [%]
  Velocity [km/h]
  Longitudinal Acceleration [m/s^2]
  Motor Torque [Nm]
  slope
  Regenerative Braking Signal

Outputs predicted at step t (from history steps t-150 to t-1):
  Velocity [km/h]
  Longitudinal Acceleration [m/s^2]
  Motor Torque [Nm]

Architecture:
  LSTM(128, return_sequences=True)  -> Dropout(0.2)
  LSTM(64, return_sequences=False)  -> Dropout(0.2)
  Dense(32, ReLU)                   -> Dropout(0.1)
  Dense(3, linear)

Run:
  python model_dynamics_lstm.py

Required files:
  Datasets/processed/vehicle_dynamics_train.csv
  Datasets/processed/vehicle_dynamics_val.csv
  Datasets/processed/vehicle_dynamics_test.csv

Outputs saved to Models/:
  dynamics_lstm_model.keras
  dynamics_lstm_feat_scaler.joblib
  dynamics_lstm_tgt_scaler.joblib
  dynamics_lstm_config.joblib
  dynamics_lstm_metrics.txt

Plots saved to Plots/:
  dynamics_lstm_training.png
  dynamics_lstm_predictions.png
  dynamics_lstm_timeseries.png
"""

import os
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import (EarlyStopping,
                                         ReduceLROnPlateau,
                                         ModelCheckpoint)
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error,
                              mean_absolute_error,
                              r2_score)

tf.random.set_seed(42)
np.random.seed(42)
tf.get_logger().setLevel('ERROR')

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
DATA_DIR  = "Datasets/processed"
MODEL_DIR = "Models"
PLOTS_DIR = "Plots"

TRAIN_FILE = os.path.join(DATA_DIR, "vehicle_dynamics_train.csv")
VAL_FILE   = os.path.join(DATA_DIR, "vehicle_dynamics_val.csv")
TEST_FILE  = os.path.join(DATA_DIR, "vehicle_dynamics_test.csv")

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
SEQ_LEN  = 150   # 150 steps = 15 seconds at 10Hz
STRIDE   = 3     # one sequence every 3 steps (3x memory reduction)

SEQ_FEATURES = [
    'Throttle [%]',
    'Velocity [km/h]',
    'Longitudinal Acceleration [m/s^2]',
    'Motor Torque [Nm]',
    'slope',
    'Regenerative Braking Signal',
]

TARGETS = [
    'Velocity [km/h]',
    'Longitudinal Acceleration [m/s^2]',
    'Motor Torque [Nm]',
]

LSTM1    = 128
LSTM2    = 64
DENSE1   = 32
DROPOUT  = 0.2
LR       = 0.001
BATCH    = 256
EPOCHS   = 50
PATIENCE = 8

TOLERANCES = {
    'Velocity [km/h]'                   : (2.0,  5.0,  'km/h'),
    'Longitudinal Acceleration [m/s^2]' : (0.3,  0.5,  'm/s2'),
    'Motor Torque [Nm]'                 : (10.0, 20.0, 'Nm'),
}

SEP = "=" * 62


# ══════════════════════════════════════════════════════════════
#  STEP 1 — LOAD
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 1 — Load processed data")
print(SEP)

for f in [TRAIN_FILE, VAL_FILE, TEST_FILE]:
    if not os.path.exists(f):
        raise FileNotFoundError(
            f"Missing: {f}\nRun preprocess_bmw_i3.py first.")

df_train = pd.read_csv(TRAIN_FILE)
df_val   = pd.read_csv(VAL_FILE)
df_test  = pd.read_csv(TEST_FILE)

print(f"  Train : {len(df_train):>10,} rows  |  "
      f"{df_train['trip_id'].nunique()} trips")
print(f"  Val   : {len(df_val):>10,} rows  |  "
      f"{df_val['trip_id'].nunique()} trips")
print(f"  Test  : {len(df_test):>10,} rows  |  "
      f"{df_test['trip_id'].nunique()} trips")

missing = [c for c in SEQ_FEATURES + TARGETS + ['trip_id']
           if c not in df_train.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")
print("  All required columns confirmed")


# ══════════════════════════════════════════════════════════════
#  STEP 2 — SCALE
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 2 — Fit scalers on training data only")
print(SEP)

feat_scaler = StandardScaler()
X_train_sc  = feat_scaler.fit_transform(df_train[SEQ_FEATURES].values)
X_val_sc    = feat_scaler.transform(df_val[SEQ_FEATURES].values)
X_test_sc   = feat_scaler.transform(df_test[SEQ_FEATURES].values)

tgt_scaler  = StandardScaler()
y_train_sc  = tgt_scaler.fit_transform(df_train[TARGETS].values)
y_val_sc    = tgt_scaler.transform(df_val[TARGETS].values)
y_test_sc   = tgt_scaler.transform(df_test[TARGETS].values)

print(f"  Feature scaler: {X_train_sc.shape[1]} features")
print(f"  Target  scaler: {y_train_sc.shape[1]} targets")


# ══════════════════════════════════════════════════════════════
#  STEP 3 — BUILD SEQUENCES
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 3 — Build LSTM sequences (within-trip only, no leakage)")
print(SEP)
print(f"  seq_len={SEQ_LEN} steps = {SEQ_LEN*0.1:.0f}s  |  "
      f"stride={STRIDE}  |  {len(SEQ_FEATURES)} features/step")


def make_sequences(X_sc, y_sc, trip_ids, seq_len, stride):
    """
    Builds sliding window sequences.

    For sequence ending at position i inside trip T:
      Input  : X_sc[i-seq_len : i]   shape (seq_len, n_features)
      Target : y_sc[i]               shape (n_targets,)

    Sequences NEVER cross trip boundaries.
    stride=3 means one sequence per every 3 eligible rows.
    """
    Xs, ys = [], []
    for tid in np.unique(trip_ids):
        mask   = (trip_ids == tid)
        X_trip = X_sc[mask]
        y_trip = y_sc[mask]
        n      = len(X_trip)
        for i in range(seq_len, n, stride):
            Xs.append(X_trip[i - seq_len : i])
            ys.append(y_trip[i])
    return (np.array(Xs, dtype=np.float32),
            np.array(ys, dtype=np.float32))


t0 = time.time()
print("  Building train sequences...")
X_tr, y_tr = make_sequences(
    X_train_sc, y_train_sc,
    df_train['trip_id'].values, SEQ_LEN, STRIDE)
print(f"    {X_tr.shape}  {X_tr.nbytes/1e6:.0f} MB  ({time.time()-t0:.1f}s)")

t0 = time.time()
print("  Building val sequences...")
X_va, y_va = make_sequences(
    X_val_sc, y_val_sc,
    df_val['trip_id'].values, SEQ_LEN, STRIDE)
print(f"    {X_va.shape}  ({time.time()-t0:.1f}s)")

t0 = time.time()
print("  Building test sequences...")
X_te, y_te = make_sequences(
    X_test_sc, y_test_sc,
    df_test['trip_id'].values, SEQ_LEN, STRIDE)
print(f"    {X_te.shape}  ({time.time()-t0:.1f}s)")


# ══════════════════════════════════════════════════════════════
#  STEP 4 — BUILD MODEL
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 4 — Build LSTM architecture")
print(SEP)

model = Sequential([
    LSTM(LSTM1,
         input_shape=(SEQ_LEN, len(SEQ_FEATURES)),
         return_sequences=True,
         name='lstm_1'),
    Dropout(DROPOUT, name='drop_1'),

    LSTM(LSTM2,
         return_sequences=False,
         name='lstm_2'),
    Dropout(DROPOUT, name='drop_2'),

    Dense(DENSE1, activation='relu', name='dense_1'),
    Dropout(0.1,   name='drop_3'),

    Dense(len(TARGETS), activation='linear', name='output'),
], name='dynamics_lstm')

model.compile(optimizer=Adam(learning_rate=LR),
              loss='mse', metrics=['mae'])
model.summary()

params = model.count_params()
print(f"\n  Parameters : {params:,}")
print(f"  Input      : (batch, {SEQ_LEN}, {len(SEQ_FEATURES)})")
print(f"  Output     : (batch, {len(TARGETS)})")


# ══════════════════════════════════════════════════════════════
#  STEP 5 — TRAIN
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 5 — Train")
print(SEP)

os.makedirs(MODEL_DIR, exist_ok=True)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=PATIENCE,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=4, min_lr=1e-6, verbose=1),
    ModelCheckpoint(os.path.join(MODEL_DIR, 'dynamics_lstm_best.keras'),
                    monitor='val_loss', save_best_only=True, verbose=0),
]

print(f"  Epochs max : {EPOCHS}  (patience={PATIENCE})")
print(f"  Batch      : {BATCH}")
print(f"  Train seqs : {len(X_tr):,}")
print(f"  Val seqs   : {len(X_va):,}\n")

t_start = time.time()
history = model.fit(
    X_tr, y_tr,
    validation_data=(X_va, y_va),
    epochs=EPOCHS,
    batch_size=BATCH,
    callbacks=callbacks,
    verbose=1,
)
t_elapsed  = time.time() - t_start
epochs_run = len(history.history['loss'])

print(f"\n  Done  {t_elapsed:.1f}s  ({t_elapsed/60:.1f} min)  "
      f"{epochs_run} epochs  "
      f"best_val_loss={min(history.history['val_loss']):.6f}")


# ══════════════════════════════════════════════════════════════
#  STEP 6 — EVALUATE
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 6 — Evaluate on test set")
print(SEP)

y_pred_sc     = model.predict(X_te, batch_size=BATCH, verbose=0)
y_test_actual = tgt_scaler.inverse_transform(y_te)
y_test_pred   = tgt_scaler.inverse_transform(y_pred_sc)

units = {
    'Velocity [km/h]'                   : 'km/h',
    'Longitudinal Acceleration [m/s^2]' : 'm/s2',
    'Motor Torque [Nm]'                 : 'Nm',
}

print(f"\n  {'Target':<42} {'RMSE':>8} {'MAE':>8} {'R2':>8}  Unit")
print(f"  {'-'*72}")
test_metrics = {}
for i, tgt in enumerate(TARGETS):
    rmse = float(np.sqrt(mean_squared_error(
                 y_test_actual[:, i], y_test_pred[:, i])))
    mae  = float(mean_absolute_error(
                 y_test_actual[:, i], y_test_pred[:, i]))
    r2   = float(r2_score(y_test_actual[:, i], y_test_pred[:, i]))
    test_metrics[tgt] = {'rmse': rmse, 'mae': mae, 'r2': r2}
    print(f"  {tgt:<42} {rmse:>8.4f} {mae:>8.4f} "
          f"{r2:>8.4f}  {units[tgt]}")

print(f"\n  Tolerance check (% of predictions within threshold):")
print(f"  {'Target':<42} {'Tight':>9} {'Wide':>9}")
print(f"  {'-'*63}")
for i, tgt in enumerate(TARGETS):
    t_val, w_val, unit = TOLERANCES[tgt]
    err   = np.abs(y_test_actual[:, i] - y_test_pred[:, i])
    pct_t = float(np.mean(err < t_val) * 100)
    pct_w = float(np.mean(err < w_val) * 100)
    print(f"  {tgt:<42} {pct_t:>8.1f}% {pct_w:>8.1f}%  "
          f"(+-{t_val} / +-{w_val} {unit})")


# ══════════════════════════════════════════════════════════════
#  STEP 7 — PLOTS
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 7 — Save diagnostic plots")
print(SEP)

os.makedirs(PLOTS_DIR, exist_ok=True)

# Plot 1: Training curves
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle('Dynamics LSTM — Training History', fontsize=12)
ax[0].plot(history.history['loss'],     label='Train')
ax[0].plot(history.history['val_loss'], label='Val')
ax[0].set_title('Loss (MSE — log scale)')
ax[0].set_yscale('log')
ax[0].legend()
ax[1].plot(history.history['mae'],     label='Train')
ax[1].plot(history.history['val_mae'], label='Val')
ax[1].set_title('MAE')
ax[1].legend()
plt.tight_layout()
p = os.path.join(PLOTS_DIR, 'dynamics_lstm_training.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {p}")

# Plot 2: Predicted vs Actual scatter
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('Dynamics LSTM — Predicted vs Actual (Test)', fontsize=12)
idx = np.random.choice(len(y_test_actual),
                        min(4000, len(y_test_actual)), replace=False)
for i, tgt in enumerate(TARGETS):
    ax[i].scatter(y_test_actual[idx, i], y_test_pred[idx, i],
                  alpha=0.2, s=5, color='steelblue')
    mn = min(y_test_actual[:, i].min(), y_test_pred[:, i].min())
    mx = max(y_test_actual[:, i].max(), y_test_pred[:, i].max())
    ax[i].plot([mn, mx], [mn, mx], 'r--', linewidth=1.5)
    lbl = tgt.split('[')[0].strip()
    ax[i].set_title(f"{lbl}\nR2={test_metrics[tgt]['r2']:.4f}  "
                    f"RMSE={test_metrics[tgt]['rmse']:.4f}")
plt.tight_layout()
p = os.path.join(PLOTS_DIR, 'dynamics_lstm_predictions.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {p}")

# Plot 3: Time-series
fig, ax = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
fig.suptitle('Dynamics LSTM — Time Series (first 600 test steps)', fontsize=12)
n_ts = min(600, len(y_test_actual))
for i, tgt in enumerate(TARGETS):
    ax[i].plot(y_test_actual[:n_ts, i], label='Actual',
               linewidth=1.3, color='steelblue')
    ax[i].plot(y_test_pred[:n_ts, i], label='Predicted',
               linewidth=1.0, linestyle='--', color='darkorange')
    ax[i].set_title(f"{tgt}  R2={test_metrics[tgt]['r2']:.4f}")
    ax[i].legend(fontsize=9)
ax[-1].set_xlabel('Step')
plt.tight_layout()
p = os.path.join(PLOTS_DIR, 'dynamics_lstm_timeseries.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {p}")


# ══════════════════════════════════════════════════════════════
#  STEP 8 — SAVE ARTEFACTS
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 8 — Save model, scalers, config, metrics")
print(SEP)

MODEL_PATH   = os.path.join(MODEL_DIR, "dynamics_lstm_model.keras")
FEAT_SC_PATH = os.path.join(MODEL_DIR, "dynamics_lstm_feat_scaler.joblib")
TGT_SC_PATH  = os.path.join(MODEL_DIR, "dynamics_lstm_tgt_scaler.joblib")
CONFIG_PATH  = os.path.join(MODEL_DIR, "dynamics_lstm_config.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "dynamics_lstm_metrics.txt")

model.save(MODEL_PATH)
joblib.dump(feat_scaler, FEAT_SC_PATH)
joblib.dump(tgt_scaler,  TGT_SC_PATH)
joblib.dump({
    'seq_len'       : SEQ_LEN,
    'seq_features'  : SEQ_FEATURES,
    'targets'       : TARGETS,
    'stride'        : STRIDE,
    'bmw_max_speed' : 150.0,
    'scooter_max'   : 25.0,
    'scale_factor'  : 25.0 / 150.0,
}, CONFIG_PATH)

with open(METRICS_PATH, 'w', encoding='utf-8') as mf:
    mf.write("Dynamics LSTM — Evaluation Metrics\n")
    mf.write("=" * 50 + "\n\n")
    mf.write(f"Architecture   : LSTM({LSTM1}) > LSTM({LSTM2}) "
             f"> Dense({DENSE1}) > Output({len(TARGETS)})\n")
    mf.write(f"Sequence length: {SEQ_LEN} steps "
             f"({SEQ_LEN * 0.1:.0f}s at 10Hz)\n")
    mf.write(f"Stride         : {STRIDE}\n")
    mf.write(f"Train seqs     : {len(X_tr):,}\n")
    mf.write(f"Training time  : {t_elapsed:.1f}s\n")
    mf.write(f"Epochs run     : {epochs_run}\n\n")
    mf.write("TEST SET RESULTS\n")
    for tgt, m in test_metrics.items():
        mf.write(f"  {tgt}: "
                 f"RMSE={m['rmse']:.4f}  "
                 f"MAE={m['mae']:.4f}  "
                 f"R2={m['r2']:.4f}\n")

print("  Models/dynamics_lstm_model.keras")
print("  Models/dynamics_lstm_feat_scaler.joblib")
print("  Models/dynamics_lstm_tgt_scaler.joblib")
print("  Models/dynamics_lstm_config.joblib")
print("  Models/dynamics_lstm_metrics.txt")


# ══════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  DYNAMICS LSTM COMPLETE")
print(SEP)
print(f"""
  Architecture : LSTM({LSTM1}) > LSTM({LSTM2}) > Dense({DENSE1}) > Output(3)
  Sequence     : {SEQ_LEN} steps = {SEQ_LEN*0.1:.0f}s of history
  Parameters   : {params:,}
  Training     : {t_elapsed:.1f}s  |  {epochs_run} epochs
""")
print(f"  {'Target':<42} {'R2':>8}  Status")
print(f"  {'-'*54}")
for tgt, m in test_metrics.items():
    st = 'PASS' if m['r2'] > 0.90 else 'WARN' if m['r2'] > 0.80 else 'FAIL'
    print(f"  {tgt:<42} {m['r2']:>8.4f}  [{st}]")
print(f"\n  Next: python model_range_lstm.py\n")
