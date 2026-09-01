"""
Motor Performance Model — LSTM
================================
Dataset  : Paderborn PMSM (processed)
           motor_train.csv / motor_val.csv / motor_test.csv

What this model does
─────────────────────
Learns the motor's electrical, mechanical, and thermal dynamics.
Given controller voltages + previous motor state, predicts the
motor's response at the next timestep.

    Input  : u_d, u_q (from controller)
             i_d_prev, i_q_prev, motor_speed_prev, omega_rad,
             pm_prev, stator_winding_prev, stator_tooth_prev,
             stator_yoke_prev, ambient, t_load
    Output : i_d, i_q, torque, motor_speed, pm, stator_winding

Architecture : LSTM (Long Short-Term Memory)
Reason       : Motor dynamics are genuinely autoregressive —
               temperature and current at time t depend on the
               full history of the motor's state. LSTM maintains
               hidden state that captures this temporal dependency
               naturally. XGBoost cannot do this correctly.

Sequence length : 50 timesteps = 25 seconds at 2Hz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR   = "Datasets/processed"
MODEL_DIR  = "Models"
PLOTS_DIR  = "Plots"

TRAIN_FILE = os.path.join(DATA_DIR, "motor_train.csv")
VAL_FILE   = os.path.join(DATA_DIR, "motor_val.csv")
TEST_FILE  = os.path.join(DATA_DIR, "motor_test.csv")

# Features and targets (must match preprocess_paderborn.py Step 6)
FEATURES = [
    'u_d',
    'u_q',
    'i_d_prev',
    'i_q_prev',
    'motor_speed_prev',
    'omega_rad',
    'pm_prev',
    'stator_winding_prev',
    'stator_tooth_prev',
    'stator_yoke_prev',
    'ambient',
    't_load',
]
TARGETS = ['i_d', 'i_q', 'torque', 'motor_speed', 'pm', 'stator_winding']

# LSTM architecture
SEQ_LEN    = 50       # 50 timesteps = 25 seconds at 2Hz
BATCH_SIZE = 512      # large batch = stable gradients on 900k rows
EPOCHS     = 50       # EarlyStopping will stop earlier if needed
PATIENCE   = 8        # stop if val_loss doesn't improve for 8 epochs

# LSTM layer sizes
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DENSE_UNITS  = 32
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001

SEP = "=" * 60
def log(msg): print(f"  {msg}")

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# ══════════════════════════════════════════════
#  STEP 1 — LOAD DATA
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 1 — Load processed data")
print(SEP)

for fpath in [TRAIN_FILE, VAL_FILE, TEST_FILE]:
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"Cannot find '{fpath}'.\n"
            f"Run preprocess_paderborn.py first."
        )

df_train = pd.read_csv(TRAIN_FILE)
df_val   = pd.read_csv(VAL_FILE)
df_test  = pd.read_csv(TEST_FILE)

log(f"Train : {len(df_train):,} rows")
log(f"Val   : {len(df_val):,} rows")
log(f"Test  : {len(df_test):,} rows")

missing = [c for c in FEATURES + TARGETS if c not in df_train.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")
log(f"All required columns present ✅")

# ══════════════════════════════════════════════
#  STEP 2 — SCALE FEATURES AND TARGETS
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 2 — Scale features and targets")
print(SEP)

# Scale features
feat_scaler = StandardScaler()
X_train_sc = feat_scaler.fit_transform(df_train[FEATURES].values)
X_val_sc   = feat_scaler.transform(df_val[FEATURES].values)
X_test_sc  = feat_scaler.transform(df_test[FEATURES].values)

# Scale targets — IMPORTANT for LSTM convergence
# We need to inverse_transform predictions later for real units
tgt_scaler = StandardScaler()
y_train_sc = tgt_scaler.fit_transform(df_train[TARGETS].values)
y_val_sc   = tgt_scaler.transform(df_val[TARGETS].values)
y_test_sc  = tgt_scaler.transform(df_test[TARGETS].values)

log(f"Feature scaler fitted on train: {X_train_sc.shape}")
log(f"Target  scaler fitted on train: {y_train_sc.shape}")

# ══════════════════════════════════════════════
#  STEP 3 — CREATE SEQUENCES
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 3 — Create LSTM sequences")
print(SEP)

def make_sequences(X, y, seq_len, profile_ids):
    """
    Create (seq_len, n_features) → (n_targets,) sequences.
    Sequences are created WITHIN each profile session only.
    No sequence spans a session boundary.
    """
    Xs, ys = [], []
    unique_pids = np.unique(profile_ids)

    for pid in unique_pids:
        mask = profile_ids == pid
        X_pid = X[mask]
        y_pid = y[mask]

        # Create sequences within this session
        for i in range(len(X_pid) - seq_len):
            Xs.append(X_pid[i : i + seq_len])
            ys.append(y_pid[i + seq_len])   # predict next timestep

    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

log(f"Building sequences (seq_len={SEQ_LEN})...")
log(f"This may take 1-2 minutes for 900k rows...")

t0 = time.time()
X_train_seq, y_train_seq = make_sequences(
    X_train_sc, y_train_sc,
    SEQ_LEN,
    df_train['profile_id'].values
)
log(f"Train sequences: {X_train_seq.shape}  ({time.time()-t0:.1f}s)")

t0 = time.time()
X_val_seq, y_val_seq = make_sequences(
    X_val_sc, y_val_sc,
    SEQ_LEN,
    df_val['profile_id'].values
)
log(f"Val   sequences: {X_val_seq.shape}  ({time.time()-t0:.1f}s)")

t0 = time.time()
X_test_seq, y_test_seq = make_sequences(
    X_test_sc, y_test_sc,
    SEQ_LEN,
    df_test['profile_id'].values
)
log(f"Test  sequences: {X_test_seq.shape}  ({time.time()-t0:.1f}s)")

# ══════════════════════════════════════════════
#  STEP 4 — BUILD LSTM MODEL
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 4 — Build LSTM architecture")
print(SEP)

model = Sequential([
    # First LSTM layer — return sequences for stacking
    LSTM(LSTM_UNITS_1,
         input_shape=(SEQ_LEN, len(FEATURES)),
         return_sequences=True,
         name='lstm_1'),
    Dropout(DROPOUT_RATE, name='dropout_1'),

    # Second LSTM layer — extract final temporal representation
    LSTM(LSTM_UNITS_2,
         return_sequences=False,
         name='lstm_2'),
    Dropout(DROPOUT_RATE, name='dropout_2'),

    # Dense layers for output mapping
    BatchNormalization(name='batch_norm'),
    Dense(DENSE_UNITS, activation='relu', name='dense_1'),
    Dropout(0.1, name='dropout_3'),

    # Output layer — one neuron per target
    Dense(len(TARGETS), activation='linear', name='output'),
])

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='mse',
    metrics=['mae']
)

model.summary()
log(f"\nInput  shape : (batch, {SEQ_LEN}, {len(FEATURES)})")
log(f"Output shape : (batch, {len(TARGETS)})")
log(f"Targets      : {TARGETS}")

# ══════════════════════════════════════════════
#  STEP 5 — TRAIN MODEL
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 5 — Train LSTM Model")
print(SEP)

os.makedirs(MODEL_DIR, exist_ok=True)
checkpoint_path = os.path.join(MODEL_DIR, "motor_model_best.keras")

callbacks = [
    # Stop training if val_loss doesn't improve for PATIENCE epochs
    EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate when plateau is hit
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    ),
    # Save best model checkpoint
    ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    ),
]

log(f"Training started...")
log(f"  Epochs     : {EPOCHS} max (EarlyStopping patience={PATIENCE})")
log(f"  Batch size : {BATCH_SIZE}")
log(f"  Train seqs : {len(X_train_seq):,}")
log(f"  Val   seqs : {len(X_val_seq):,}")
log("")

t_start = time.time()
history = model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)
t_elapsed = time.time() - t_start
epochs_run = len(history.history['loss'])

log(f"\nTraining complete in {t_elapsed:.1f} seconds")
log(f"Epochs run   : {epochs_run}")
log(f"Best val_loss: {min(history.history['val_loss']):.6f}")

# ══════════════════════════════════════════════
#  STEP 6 — EVALUATE ON TEST SET
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 6 — Evaluate on test set")
print(SEP)

y_test_pred_sc = model.predict(X_test_seq, batch_size=BATCH_SIZE, verbose=0)

# Inverse transform back to original units
y_test_actual = tgt_scaler.inverse_transform(y_test_seq)
y_test_pred   = tgt_scaler.inverse_transform(y_test_pred_sc)

log(f"{'Target':<20} {'RMSE':>10} {'MAE':>10} {'R²':>10}  {'Unit'}")
log(f"{'-'*65}")

units = {
    'i_d'           : 'A',
    'i_q'           : 'A',
    'torque'        : 'Nm',
    'motor_speed'   : 'RPM',
    'pm'            : '°C',
    'stator_winding': '°C',
}

test_metrics = {}
for i, target in enumerate(TARGETS):
    rmse = np.sqrt(mean_squared_error(y_test_actual[:, i], y_test_pred[:, i]))
    mae  = mean_absolute_error(y_test_actual[:, i], y_test_pred[:, i])
    r2   = r2_score(y_test_actual[:, i], y_test_pred[:, i])
    test_metrics[target] = {'rmse': rmse, 'mae': mae, 'r2': r2}
    unit = units.get(target, '')
    log(f"  {target:<18} {rmse:>10.4f} {mae:>10.4f} {r2:>10.4f}  {unit}")

print()
log("R² Score Guide:")
log("  R² > 0.95 → Excellent  ✅")
log("  R² > 0.90 → Good       ✅")
log("  R² > 0.80 → Acceptable ⚠️")
log("  R² < 0.80 → Needs work ❌")

# ══════════════════════════════════════════════
#  STEP 7 — SAVE DIAGNOSTIC PLOTS
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 7 — Save diagnostic plots")
print(SEP)

os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Plot 1: Training history ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Motor Performance LSTM — Training History', fontsize=13)

axes[0].plot(history.history['loss'],     label='Train Loss', color='steelblue')
axes[0].plot(history.history['val_loss'], label='Val Loss',   color='darkorange')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE Loss')
axes[0].set_title('Loss Curve')
axes[0].legend()
axes[0].set_yscale('log')

axes[1].plot(history.history['mae'],     label='Train MAE', color='steelblue')
axes[1].plot(history.history['val_mae'], label='Val MAE',   color='darkorange')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('MAE')
axes[1].set_title('MAE Curve')
axes[1].legend()

plt.tight_layout()
hist_path = os.path.join(PLOTS_DIR, 'motor_training_history.png')
plt.savefig(hist_path, dpi=150, bbox_inches='tight')
plt.close()
log(f"Saved: {hist_path}")

# ── Plot 2: Predicted vs Actual for all 6 targets ──
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Motor Performance LSTM — Predictions vs Actual (Test Set)', fontsize=13)
axes = axes.flatten()

n_plot = min(3000, len(y_test_actual))
idx    = np.random.choice(len(y_test_actual), n_plot, replace=False)

for i, target in enumerate(TARGETS):
    ax = axes[i]
    ax.scatter(y_test_actual[idx, i], y_test_pred[idx, i],
               alpha=0.3, s=5, color='steelblue')
    mn = min(y_test_actual[:, i].min(), y_test_pred[:, i].min())
    mx = max(y_test_actual[:, i].max(), y_test_pred[:, i].max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect fit')
    r2   = test_metrics[target]['r2']
    rmse = test_metrics[target]['rmse']
    unit = units.get(target, '')
    ax.set_xlabel(f'Actual {target} [{unit}]')
    ax.set_ylabel(f'Predicted {target} [{unit}]')
    ax.set_title(f'{target}  R²={r2:.4f}  RMSE={rmse:.3f}')
    ax.legend(fontsize=8)

plt.tight_layout()
pred_path = os.path.join(PLOTS_DIR, 'motor_predictions_vs_actual.png')
plt.savefig(pred_path, dpi=150, bbox_inches='tight')
plt.close()
log(f"Saved: {pred_path}")

# ── Plot 3: Time-series prediction for one test session ──
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('Motor Performance LSTM — Time-Series Prediction (Test Sample)', fontsize=13)
axes = axes.flatten()

n_ts = min(1000, len(y_test_actual))
for i, target in enumerate(TARGETS):
    ax = axes[i]
    ax.plot(y_test_actual[:n_ts, i], label='Actual',    color='steelblue',  linewidth=1.2)
    ax.plot(y_test_pred[:n_ts, i],   label='Predicted', color='darkorange', linewidth=1.0, linestyle='--')
    unit = units.get(target, '')
    ax.set_xlabel('Timestep')
    ax.set_ylabel(f'{target} [{unit}]')
    r2 = test_metrics[target]['r2']
    ax.set_title(f'{target}  R²={r2:.4f}')
    ax.legend(fontsize=8)

plt.tight_layout()
ts_path = os.path.join(PLOTS_DIR, 'motor_timeseries_prediction.png')
plt.savefig(ts_path, dpi=150, bbox_inches='tight')
plt.close()
log(f"Saved: {ts_path}")

# ══════════════════════════════════════════════
#  STEP 8 — SAVE MODEL AND SCALERS
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 8 — Save model and scalers")
print(SEP)

# Save full model in Keras format
model_path       = os.path.join(MODEL_DIR, "motor_model.keras")
feat_scaler_path = os.path.join(MODEL_DIR, "motor_feat_scaler.joblib")
tgt_scaler_path  = os.path.join(MODEL_DIR, "motor_tgt_scaler.joblib")

model.save(model_path)
joblib.dump(feat_scaler, feat_scaler_path)
joblib.dump(tgt_scaler,  tgt_scaler_path)

log(f"Saved model        : {model_path}")
log(f"Saved feat scaler  : {feat_scaler_path}")
log(f"Saved tgt  scaler  : {tgt_scaler_path}")
log(f"Saved best ckpt    : {checkpoint_path}")

# Save metrics
metrics_path = os.path.join(MODEL_DIR, "motor_metrics.txt")
with open(metrics_path, 'w') as f:
    f.write("Motor Performance Model — Evaluation Metrics\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Architecture: LSTM ({LSTM_UNITS_1}) → LSTM ({LSTM_UNITS_2}) → Dense ({DENSE_UNITS}) → Output ({len(TARGETS)})\n")
    f.write(f"Sequence length: {SEQ_LEN} timesteps ({SEQ_LEN * 0.5:.1f} seconds)\n")
    f.write(f"Training time: {t_elapsed:.1f} seconds\n")
    f.write(f"Epochs run: {epochs_run}\n\n")
    f.write("TEST SET RESULTS\n")
    for target, m in test_metrics.items():
        unit = units.get(target, '')
        f.write(f"  {target:<20}: RMSE={m['rmse']:.4f} {unit}, MAE={m['mae']:.4f} {unit}, R²={m['r2']:.4f}\n")
    f.write(f"\nFeatures used ({len(FEATURES)}): {FEATURES}\n")
    f.write(f"Targets  ({len(TARGETS)}): {TARGETS}\n")

log(f"Saved metrics      : {metrics_path}")

# Save sequence config for Digital Twin inference
config_path = os.path.join(MODEL_DIR, "motor_config.joblib")
config = {
    'seq_len'  : SEQ_LEN,
    'features' : FEATURES,
    'targets'  : TARGETS,
}
joblib.dump(config, config_path)
log(f"Saved config       : {config_path}")

# ══════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  MOTOR MODEL COMPLETE")
print(SEP)
print(f"""
  Model      : LSTM ({LSTM_UNITS_1}) → LSTM ({LSTM_UNITS_2}) → Dense → Output
  Seq length : {SEQ_LEN} timesteps ({SEQ_LEN * 0.5:.1f} seconds context)
  Features   : {len(FEATURES)}
  Targets    : {len(TARGETS)}  →  {TARGETS}
  Train time : {t_elapsed:.1f} seconds
  Epochs     : {epochs_run}

  TEST RESULTS:
""")
for target, m in test_metrics.items():
    unit   = units.get(target, '')
    status = "✅" if m['r2'] > 0.95 else ("✅" if m['r2'] > 0.90 else ("⚠️" if m['r2'] > 0.80 else "❌"))
    print(f"    {target:<20} R²={m['r2']:.4f}  RMSE={m['rmse']:.4f} {unit}  {status}")

print(f"""
  Saved to {MODEL_DIR}/
    motor_model.keras          ← load this for Digital Twin
    motor_feat_scaler.joblib   ← apply to input features
    motor_tgt_scaler.joblib    ← inverse transform predictions
    motor_config.joblib        ← seq_len and feature list
    motor_metrics.txt          ← full metrics report

  Next step:
    Run model_vehicle_dynamics.py  (XGBoost on BMW i3)
""")
