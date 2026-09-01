"""
LSTM Range Management Model
=============================
Replaces XGBoost Range Management model.
Uses 200 timesteps (20 seconds) of battery and driving history
to predict TWO outputs simultaneously:

  Output 1: SoC [%]        (Choice A — state of charge at next step)
  Output 2: remaining_km   (Choice B — estimated range left in km)

Why both outputs?
  SoC alone tells you battery level but not how far you can go.
  remaining_km alone hides the underlying battery state.
  Together they give the complete picture:
    SoC tells the DASHBOARD how full the battery is.
    remaining_km tells the RIDER how far they can still travel.

Why LSTM for range?
  XGBoost sees only SoC(t-1) — one step of history.
  LSTM sees 200 steps (20 seconds) of full battery drain pattern.
  This lets the model learn:
    - The RATE at which SoC is falling (trend)
    - Whether drain is accelerating (uphill trend)
    - Whether drain is slowing (recovering from hill)
    - How battery temperature affects drain
  All of which are invisible to a single-step XGBoost predictor.

SoC monotonicity:
  Real SoC can only decrease during discharge.
  It can SLIGHTLY increase during regenerative braking.
  The model learns this from data.
  Post-prediction enforcement is applied in models.py:
    if regen == 0: SoC(t) = min(predicted_SoC, SoC(t-1))
    if regen > 0:  SoC(t) = min(predicted_SoC, SoC(t-1) + 0.5)

remaining_km label derivation (Choice B ground truth):
  For each trip, the ground truth is derived using trip-level efficiency:
    total_energy_Wh  = sum(energy_norm_Whkg * BMW_MASS * dt)
    total_dist_km    = sum(Velocity / 3.6 * dt)
    trip_eff_Whkm    = total_energy_Wh / total_dist_km  (BMW i3)
    scooter_eff_Whkm = trip_eff_Whkm * (SCOOTER_MASS / BMW_MASS)
    remaining_km[t]  = (SoC[t] / 100 * SCOOTER_BAT_WH) / scooter_eff_Whkm

  WHY trip-level efficiency (not rolling)?
    Rolling efficiency (cum_wh/cum_km) is noisy at trip start
    (tiny distance denominator). Trip-level uses the full trip
    as stable ground truth. The LSTM learns to approximate
    this from the SoC drain pattern it observes in its 200-step window.

  WHY NOT simple distance subtraction?
    remaining_km != total_range - distance_covered
    Because total_range is not constant — it depends on
    slope, throttle, temperature throughout the trip.
    A rider climbing a hill has less remaining range than
    one on flat road, even at the same SoC and same distance covered.
    The LSTM learns this because it sees slope and throttle history.

Sequence design:
  Length  : 200 steps = 20 seconds at 10Hz
  Stride  : 3  (one sequence every 3 steps)
  Features: 6 per step (battery state + driving context)

Sequence input features (what each of the 200 steps contains):
  SoC [%]                    <- battery state history (most important)
  energy_norm_Whkg           <- energy consumption rate history
  Velocity [km/h]            <- speed history (distance context)
  Throttle [%]               <- load history
  slope                      <- gradient context
  Battery Temperature [°C]   <- thermal history

Outputs predicted at step t (from history t-200 to t-1):
  SoC [%]         (Choice A)
  remaining_km    (Choice B)

Architecture:
  LSTM(128, return_sequences=True)  -> Dropout(0.2)
  LSTM(64, return_sequences=False)  -> Dropout(0.2)
  Dense(32, ReLU)                   -> Dropout(0.1)
  Dense(2, linear)   [SoC, remaining_km]

Run:
  python model_range_lstm.py

Required files:
  Datasets/processed/range_management_train.csv
  Datasets/processed/range_management_val.csv
  Datasets/processed/range_management_test.csv

Outputs saved to Models/:
  range_lstm_model.keras
  range_lstm_feat_scaler.joblib
  range_lstm_tgt_scaler.joblib
  range_lstm_config.joblib
  range_lstm_metrics.txt

Plots saved to Plots/:
  range_lstm_training.png
  range_lstm_predictions.png
  range_lstm_timeseries.png
  range_lstm_monotonicity.png
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
#  PHYSICAL CONSTANTS
# ─────────────────────────────────────────────
SCOOTER_MASS   = 90.0       # kg
BMW_MASS       = 1270.0     # kg (training data vehicle)
SCOOTER_BAT_WH = 446.0      # Wh (e-scooter battery)
DT             = 0.1        # seconds per step at 10Hz

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
DATA_DIR  = "Datasets/processed"
MODEL_DIR = "Models"
PLOTS_DIR = "Plots"

TRAIN_FILE = os.path.join(DATA_DIR, "range_management_train.csv")
VAL_FILE   = os.path.join(DATA_DIR, "range_management_val.csv")
TEST_FILE  = os.path.join(DATA_DIR, "range_management_test.csv")

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
SEQ_LEN  = 200   # 200 steps = 20 seconds at 10Hz
STRIDE   = 3     # one sequence every 3 steps

# Sequence features — 6 temporal signals per step
# These capture the full battery drain and driving context history
SEQ_FEATURES = [
    'SoC [%]',                    # battery state — most critical
    'energy_norm_Whkg',           # energy consumption rate
    'Velocity [km/h]',            # speed context
    'Throttle [%]',               # load context
    'slope',                      # gradient context
    'Battery Temperature [°C]',   # thermal context
]

# Two outputs — Choice A and Choice B combined
TARGETS = [
    'SoC [%]',        # Choice A: state of charge at next step
    'remaining_km',   # Choice B: estimated range remaining
]

LSTM1    = 128
LSTM2    = 64
DENSE1   = 32
DROPOUT  = 0.2
LR       = 0.001
BATCH    = 256
EPOCHS   = 50
PATIENCE = 8

# Validation tolerances
TOLERANCES = {
    'SoC [%]'      : (0.5,  1.0,  '%'),
    'remaining_km' : (2.0,  5.0,  'km'),
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

# Confirm sequence features are present
missing_seq = [c for c in SEQ_FEATURES + ['SoC [%]', 'energy_norm_Whkg', 'trip_id']
               if c not in df_train.columns]
if missing_seq:
    raise ValueError(f"Missing columns in training data: {missing_seq}")
print("  All sequence feature columns confirmed")


# ══════════════════════════════════════════════════════════════
#  STEP 2 — DERIVE remaining_km LABEL (Choice B ground truth)
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 2 — Derive remaining_km label for each trip")
print(SEP)
print("""
  Method: Trip-level energy efficiency
    For each trip:
      total_energy_Wh  = sum(|energy_norm_Whkg| * BMW_MASS * DT)
      total_dist_km    = sum(Velocity / 3.6 * DT)
      trip_eff_Whkm    = total_energy_Wh / total_dist_km  (BMW scale)
      scooter_eff      = trip_eff_Whkm * (SCOOTER_MASS / BMW_MASS)
      remaining_km[t]  = (SoC[t] / 100 * SCOOTER_BAT_WH) / scooter_eff

  Why trip-level (not rolling)?
    Rolling efficiency is noisy at trip start due to tiny distance
    denominator. Trip-level provides a stable, consistent ground
    truth that the LSTM learns to approximate from history.

  Why NOT distance subtraction?
    remaining_km != total_range - distance_covered
    Because total_range varies with slope, throttle, temperature.
    This approach correctly uses SoC and actual energy efficiency.
""")


def derive_remaining_km(df, scooter_bat_wh, scooter_mass, bmw_mass, dt):
    """
    Adds 'remaining_km' column to dataframe.
    Computed per trip using trip-level efficiency as ground truth.
    """
    df = df.copy()
    df['remaining_km'] = np.nan

    for tid in df['trip_id'].unique():
        mask  = df['trip_id'] == tid
        trip  = df[mask].copy()

        # Total energy consumed in BMW i3 across entire trip
        total_energy_wh = (np.abs(trip['energy_norm_Whkg']) * bmw_mass * dt).sum()
        # Total distance in km
        total_dist_km   = (trip['Velocity [km/h]'] / 3.6 * dt / 1000).sum()

        if total_dist_km < 0.1:
            # Too short a trip to derive meaningful efficiency
            # Use a conservative fallback
            scooter_eff = (134.0 * scooter_mass / bmw_mass)
        else:
            trip_eff_whkm   = total_energy_wh / total_dist_km
            scooter_eff     = trip_eff_whkm * (scooter_mass / bmw_mass)

        # Clip efficiency to physically meaningful range for an e-scooter
        # Below 5 Wh/km is unrealistic, above 50 Wh/km is extreme hill climbing
        scooter_eff = float(np.clip(scooter_eff, 5.0, 50.0))

        # remaining_km at each step = remaining energy / efficiency
        remaining = (trip['SoC [%]'] / 100.0 * scooter_bat_wh) / scooter_eff
        df.loc[mask, 'remaining_km'] = remaining.clip(0, 100).values

    return df


print("  Deriving labels for train set...")
t0 = time.time()
df_train = derive_remaining_km(df_train, SCOOTER_BAT_WH, SCOOTER_MASS, BMW_MASS, DT)
print(f"    Done  {time.time()-t0:.1f}s  "
      f"remaining_km range: "
      f"{df_train['remaining_km'].min():.1f} -- "
      f"{df_train['remaining_km'].max():.1f} km")

print("  Deriving labels for val set...")
t0 = time.time()
df_val = derive_remaining_km(df_val, SCOOTER_BAT_WH, SCOOTER_MASS, BMW_MASS, DT)
print(f"    Done  {time.time()-t0:.1f}s  "
      f"remaining_km range: "
      f"{df_val['remaining_km'].min():.1f} -- "
      f"{df_val['remaining_km'].max():.1f} km")

print("  Deriving labels for test set...")
t0 = time.time()
df_test = derive_remaining_km(df_test, SCOOTER_BAT_WH, SCOOTER_MASS, BMW_MASS, DT)
print(f"    Done  {time.time()-t0:.1f}s  "
      f"remaining_km range: "
      f"{df_test['remaining_km'].min():.1f} -- "
      f"{df_test['remaining_km'].max():.1f} km")

# Verify label makes physical sense
soc_start  = df_train['SoC [%]'].iloc[0]
rem_start  = df_train['remaining_km'].iloc[0]
print(f"\n  Verification (first train row):")
print(f"    SoC={soc_start:.1f}%  ->  "
      f"remaining_km={rem_start:.1f} km  (expected ~25-50 km)")


# ══════════════════════════════════════════════════════════════
#  STEP 3 — SCALE
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 3 — Fit scalers on training data only")
print(SEP)

feat_scaler = StandardScaler()
X_train_sc  = feat_scaler.fit_transform(df_train[SEQ_FEATURES].values)
X_val_sc    = feat_scaler.transform(df_val[SEQ_FEATURES].values)
X_test_sc   = feat_scaler.transform(df_test[SEQ_FEATURES].values)

tgt_scaler  = StandardScaler()
y_train_sc  = tgt_scaler.fit_transform(df_train[TARGETS].values)
y_val_sc    = tgt_scaler.transform(df_val[TARGETS].values)
y_test_sc   = tgt_scaler.transform(df_test[TARGETS].values)

print(f"  Feature scaler: {X_train_sc.shape[1]} features  "
      f"(SoC mean={feat_scaler.mean_[0]:.1f}%, "
      f"remaining_km mean={tgt_scaler.mean_[1]:.1f} km)")
print(f"  Target  scaler: {y_train_sc.shape[1]} targets  "
      f"[SoC, remaining_km]")


# ══════════════════════════════════════════════════════════════
#  STEP 4 — BUILD SEQUENCES
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 4 — Build LSTM sequences (within-trip only, no leakage)")
print(SEP)
print(f"  seq_len={SEQ_LEN} steps = {SEQ_LEN*0.1:.0f}s  |  "
      f"stride={STRIDE}  |  {len(SEQ_FEATURES)} features/step")
print(f"  Sequence input captures: SoC drain trend, energy rate,")
print(f"  speed history, throttle pattern, slope, battery temp")


def make_sequences(X_sc, y_sc, trip_ids, seq_len, stride):
    """
    Builds sliding window sequences within each trip.

    For sequence ending at position i inside trip T:
      Input  : X_sc[i - seq_len : i]   shape (seq_len, n_features)
      Target : y_sc[i]                 shape (2,) = [SoC, remaining_km]

    Key design: the input sequence INCLUDES SoC history.
    So the model sees: how SoC fell over the last 20 seconds.
    From this pattern it predicts: next SoC AND remaining range.

    Sequences NEVER cross trip boundaries — enforced by the
    per-trip loop. This prevents the model from seeing end
    of Trip A as the start of Trip B.
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
#  STEP 5 — BUILD MODEL
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 5 — Build LSTM architecture")
print(SEP)

model = Sequential([
    # Layer 1: learns short-term patterns
    # (sudden throttle changes, rapid SoC drops on hard acceleration)
    LSTM(LSTM1,
         input_shape=(SEQ_LEN, len(SEQ_FEATURES)),
         return_sequences=True,
         name='lstm_1'),
    Dropout(DROPOUT, name='drop_1'),

    # Layer 2: compresses the 200-step SoC drain history
    # into a single context vector capturing the drain TREND
    LSTM(LSTM2,
         return_sequences=False,
         name='lstm_2'),
    Dropout(DROPOUT, name='drop_2'),

    # Dense head — projects context to both outputs
    Dense(DENSE1, activation='relu', name='dense_1'),
    Dropout(0.1, name='drop_3'),

    # Dual output head — predicts SoC AND remaining_km simultaneously
    # Both trained together — the model learns their relationship
    # (remaining_km is directly tied to SoC through efficiency)
    Dense(len(TARGETS), activation='linear', name='output'),
], name='range_lstm')

model.compile(
    optimizer=Adam(learning_rate=LR),
    loss='mse',
    metrics=['mae'],
)
model.summary()

params = model.count_params()
print(f"\n  Parameters  : {params:,}")
print(f"  Input       : (batch, {SEQ_LEN}, {len(SEQ_FEATURES)})")
print(f"  Output      : (batch, 2)  [SoC, remaining_km]")
print(f"  Output[0]   : SoC [%]         <- Choice A")
print(f"  Output[1]   : remaining_km    <- Choice B")


# ══════════════════════════════════════════════════════════════
#  STEP 6 — TRAIN
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 6 — Train")
print(SEP)

os.makedirs(MODEL_DIR, exist_ok=True)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=PATIENCE,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=4, min_lr=1e-6, verbose=1),
    ModelCheckpoint(os.path.join(MODEL_DIR, 'range_lstm_best.keras'),
                    monitor='val_loss', save_best_only=True, verbose=0),
]

print(f"  Max epochs : {EPOCHS}  (patience={PATIENCE})")
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
#  STEP 7 — EVALUATE
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 7 — Evaluate on test set")
print(SEP)

y_pred_sc     = model.predict(X_te, batch_size=BATCH, verbose=0)
y_test_actual = tgt_scaler.inverse_transform(y_te)
y_test_pred   = tgt_scaler.inverse_transform(y_pred_sc)

print(f"\n  {'Target':<18} {'RMSE':>8} {'MAE':>8} {'R2':>8}  Unit")
print(f"  {'-'*50}")
test_metrics = {}
for i, tgt in enumerate(TARGETS):
    rmse = float(np.sqrt(mean_squared_error(
                 y_test_actual[:, i], y_test_pred[:, i])))
    mae  = float(mean_absolute_error(
                 y_test_actual[:, i], y_test_pred[:, i]))
    r2   = float(r2_score(y_test_actual[:, i], y_test_pred[:, i]))
    unit = '%' if tgt == 'SoC [%]' else 'km'
    test_metrics[tgt] = {'rmse': rmse, 'mae': mae, 'r2': r2}
    print(f"  {tgt:<18} {rmse:>8.4f} {mae:>8.4f} {r2:>8.4f}  {unit}")

print(f"\n  Tolerance check:")
print(f"  {'Target':<18} {'Tight':>9} {'Wide':>9}")
print(f"  {'-'*40}")
for i, tgt in enumerate(TARGETS):
    t_val, w_val, unit = TOLERANCES[tgt]
    err   = np.abs(y_test_actual[:, i] - y_test_pred[:, i])
    pct_t = float(np.mean(err < t_val) * 100)
    pct_w = float(np.mean(err < w_val) * 100)
    print(f"  {tgt:<18} {pct_t:>8.1f}% {pct_w:>8.1f}%  "
          f"(+-{t_val} / +-{w_val} {unit})")

# Monotonicity check on SoC predictions
# SoC should be non-increasing when no regen applied
# We test this on the test sequences: SoC should not jump up randomly
print(f"\n  SoC monotonicity analysis:")
soc_pred = y_test_pred[:, 0]
soc_actual = y_test_actual[:, 0]
# Check how often prediction is higher than previous actual
# (not perfectly testable here but directionally informative)
soc_pred_diff = np.diff(soc_pred)
pct_increase  = float(np.mean(soc_pred_diff > 0.5) * 100)
pct_flat_down = float(np.mean(soc_pred_diff <= 0.5) * 100)
print(f"    Prediction increases >0.5%: {pct_increase:.1f}%  "
      f"(ideally low — enforced in dashboard)")
print(f"    Prediction flat/decreasing: {pct_flat_down:.1f}%")
print(f"    Note: Hard monotonicity enforcement applied in models.py")


# ══════════════════════════════════════════════════════════════
#  STEP 8 — PLOTS
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 8 — Save diagnostic plots")
print(SEP)

os.makedirs(PLOTS_DIR, exist_ok=True)

# Plot 1: Training history
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle('Range LSTM — Training History', fontsize=12)
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
p = os.path.join(PLOTS_DIR, 'range_lstm_training.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {p}")

# Plot 2: Predicted vs Actual scatter for both outputs
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Range LSTM — Predicted vs Actual (Test)', fontsize=12)
idx = np.random.choice(len(y_test_actual),
                        min(4000, len(y_test_actual)), replace=False)
labels = ['SoC [%] (Choice A)', 'Remaining Range km (Choice B)']
for i in range(2):
    ax[i].scatter(y_test_actual[idx, i], y_test_pred[idx, i],
                  alpha=0.2, s=5, color='steelblue')
    mn = min(y_test_actual[:, i].min(), y_test_pred[:, i].min())
    mx = max(y_test_actual[:, i].max(), y_test_pred[:, i].max())
    ax[i].plot([mn, mx], [mn, mx], 'r--', linewidth=1.5)
    ax[i].set_xlabel('Actual')
    ax[i].set_ylabel('Predicted')
    r2   = test_metrics[TARGETS[i]]['r2']
    rmse = test_metrics[TARGETS[i]]['rmse']
    ax[i].set_title(f"{labels[i]}\nR2={r2:.4f}  RMSE={rmse:.4f}")
plt.tight_layout()
p = os.path.join(PLOTS_DIR, 'range_lstm_predictions.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {p}")

# Plot 3: Time series for both outputs
fig, ax = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
fig.suptitle('Range LSTM — Time Series (first 600 test steps)', fontsize=12)
n_ts = min(600, len(y_test_actual))
for i in range(2):
    ax[i].plot(y_test_actual[:n_ts, i], label='Actual',
               linewidth=1.3, color='steelblue')
    ax[i].plot(y_test_pred[:n_ts, i], label='Predicted',
               linewidth=1.0, linestyle='--', color='darkorange')
    ax[i].set_title(f"{TARGETS[i]}  "
                    f"R2={test_metrics[TARGETS[i]]['r2']:.4f}  "
                    f"RMSE={test_metrics[TARGETS[i]]['rmse']:.4f}")
    ax[i].legend(fontsize=9)
ax[-1].set_xlabel('Step')
plt.tight_layout()
p = os.path.join(PLOTS_DIR, 'range_lstm_timeseries.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {p}")

# Plot 4: SoC monotonicity visualization
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
fig.suptitle('Range LSTM — SoC Prediction Direction Analysis', fontsize=12)
n_mono = min(500, len(y_test_actual))
ax.plot(y_test_actual[:n_mono, 0], label='Actual SoC',
        color='steelblue', linewidth=1.3)
ax.plot(y_test_pred[:n_mono, 0], label='Predicted SoC',
        color='darkorange', linewidth=1.0, linestyle='--')
ax.set_xlabel('Step')
ax.set_ylabel('SoC [%]')
ax.set_title(
    'SoC should be non-increasing (monotonic) during discharge\n'
    'Hard enforcement applied in models.py: SoC(t) = min(predicted, SoC_prev)')
ax.legend()
plt.tight_layout()
p = os.path.join(PLOTS_DIR, 'range_lstm_monotonicity.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print(f"  Saved: {p}")


# ══════════════════════════════════════════════════════════════
#  STEP 9 — SAVE ARTEFACTS
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 9 — Save model, scalers, config, metrics")
print(SEP)

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH   = os.path.join(MODEL_DIR, "range_lstm_model.keras")
FEAT_SC_PATH = os.path.join(MODEL_DIR, "range_lstm_feat_scaler.joblib")
TGT_SC_PATH  = os.path.join(MODEL_DIR, "range_lstm_tgt_scaler.joblib")
CONFIG_PATH  = os.path.join(MODEL_DIR, "range_lstm_config.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "range_lstm_metrics.txt")

model.save(MODEL_PATH)
joblib.dump(feat_scaler, FEAT_SC_PATH)
joblib.dump(tgt_scaler,  TGT_SC_PATH)
joblib.dump({
    'seq_len'         : SEQ_LEN,
    'seq_features'    : SEQ_FEATURES,
    'targets'         : TARGETS,
    'stride'          : STRIDE,
    # Physical constants used for label derivation
    # (needed by models.py for inference)
    'scooter_mass'    : SCOOTER_MASS,
    'bmw_mass'        : BMW_MASS,
    'scooter_bat_wh'  : SCOOTER_BAT_WH,
    # Output indices
    'soc_idx'         : 0,
    'remaining_km_idx': 1,
}, CONFIG_PATH)

with open(METRICS_PATH, 'w', encoding='utf-8') as mf:
    mf.write("Range LSTM — Evaluation Metrics\n")
    mf.write("=" * 50 + "\n\n")
    mf.write(f"Architecture   : LSTM({LSTM1}) > LSTM({LSTM2}) "
             f"> Dense({DENSE1}) > Output(2)\n")
    mf.write(f"Sequence length: {SEQ_LEN} steps "
             f"({SEQ_LEN * 0.1:.0f}s at 10Hz)\n")
    mf.write(f"Stride         : {STRIDE}\n")
    mf.write(f"Train seqs     : {len(X_tr):,}\n")
    mf.write(f"Training time  : {t_elapsed:.1f}s\n")
    mf.write(f"Epochs run     : {epochs_run}\n\n")
    mf.write("Outputs:\n")
    mf.write("  Output[0] = SoC [%]         (Choice A)\n")
    mf.write("  Output[1] = remaining_km    (Choice B)\n\n")
    mf.write("Label derivation:\n")
    mf.write("  remaining_km[t] = (SoC[t]/100 * 446 Wh) / scooter_eff_Whkm\n")
    mf.write("  scooter_eff = trip_total_energy * (90kg/1270kg) "
             "/ trip_total_dist\n\n")
    mf.write("TEST SET RESULTS\n")
    mf.write("-" * 50 + "\n")
    for tgt, m in test_metrics.items():
        mf.write(f"  {tgt}: "
                 f"RMSE={m['rmse']:.4f}  "
                 f"MAE={m['mae']:.4f}  "
                 f"R2={m['r2']:.4f}\n")

print("  Models/range_lstm_model.keras")
print("  Models/range_lstm_feat_scaler.joblib")
print("  Models/range_lstm_tgt_scaler.joblib")
print("  Models/range_lstm_config.joblib")
print("  Models/range_lstm_metrics.txt")


# ══════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("  RANGE LSTM COMPLETE")
print(SEP)
print(f"""
  Architecture : LSTM({LSTM1}) > LSTM({LSTM2}) > Dense({DENSE1}) > Output(2)
  Sequence     : {SEQ_LEN} steps = {SEQ_LEN*0.1:.0f}s of battery drain history
  Parameters   : {params:,}
  Training     : {t_elapsed:.1f}s  |  {epochs_run} epochs

  Outputs:
    [0] SoC [%]        <- current battery level (Choice A)
    [1] remaining_km   <- estimated range left   (Choice B)

  Label method:
    remaining_km derived from trip-level efficiency per BMW i3 trip
    scaled to e-scooter via mass ratio (90/1270 = 0.071)
    NOT from distance subtraction
""")
print(f"  {'Target':<20} {'R2':>8}  {'RMSE':>8}  Status")
print(f"  {'-'*44}")
for tgt, m in test_metrics.items():
    unit = '%' if tgt == 'SoC [%]' else 'km'
    st   = 'PASS' if m['r2'] > 0.90 else 'WARN' if m['r2'] > 0.80 else 'FAIL'
    print(f"  {tgt:<20} {m['r2']:>8.4f}  "
          f"{m['rmse']:>8.4f} {unit}  [{st}]")

print(f"""
  SoC monotonicity:
    Enforced post-prediction in models.py
    SoC(t) = min(predicted, SoC_prev)      when regen == 0
    SoC(t) = min(predicted, SoC_prev+0.5)  when regen > 0

  Next step:
    Update models.py to load LSTM models instead of XGBoost
    Run: python update_models_for_lstm.py
""")
