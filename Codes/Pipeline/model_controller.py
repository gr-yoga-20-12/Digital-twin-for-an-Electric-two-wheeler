"""
Motor Controller Model — XGBoost
==================================
Dataset  : Paderborn PMSM (processed)
           motor_train.csv / motor_val.csv / motor_test.csv

What this model does
─────────────────────
Learns how the controller generates phase voltages (u_d, u_q)
from throttle command + motor feedback state.

    Input  : throttle_proxy, i_d_prev, i_q_prev,
             motor_speed_prev, pm_prev, stator_winding_prev,
             ambient, duty_cycle_prev
    Output : u_d, u_q  (d/q axis phase voltages)

In the Digital Twin feedback loop:
    Rider throttle → Controller Model → u_d, u_q → Motor Model

Architecture : XGBoost (MultiOutputRegressor wrapping two trees)
Reason       : Controller is a stateless mapping function,
               not a temporal dynamics problem. XGBoost handles
               this cleanly and trains in minutes on 900k rows.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import time
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
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
    'throttle_proxy',
    'i_d_prev',
    'i_q_prev',
    'motor_speed_prev',
    'pm_prev',
    'stator_winding_prev',
    'ambient',
    'duty_cycle_prev',
]
TARGETS = ['u_d', 'u_q']

# XGBoost hyperparameters
# Tuned for 900k rows — balanced between accuracy and training time
XGB_PARAMS = {
    'n_estimators'     : 500,
    'max_depth'        : 6,
    'learning_rate'    : 0.05,
    'subsample'        : 0.8,
    'colsample_bytree' : 0.8,
    'reg_lambda'       : 2.0,       # L2 regularisation — prevents overfitting
    'reg_alpha'        : 0.1,       # L1 regularisation
    'min_child_weight' : 5,
    'random_state'     : 42,
    'n_jobs'           : -1,        # use all CPU cores
    'tree_method'      : 'hist',    # fast histogram method
    'verbosity'        : 0,
}

SEP = "=" * 60
def log(msg): print(f"  {msg}")

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

# Verify all required columns exist
all_cols = FEATURES + TARGETS
missing = [c for c in all_cols if c not in df_train.columns]
if missing:
    raise ValueError(f"Missing columns in training data: {missing}")
log(f"All required columns present ✅")

# ══════════════════════════════════════════════
#  STEP 2 — PREPARE FEATURES AND TARGETS
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 2 — Prepare features and targets")
print(SEP)

X_train = df_train[FEATURES].values
y_train = df_train[TARGETS].values

X_val   = df_val[FEATURES].values
y_val   = df_val[TARGETS].values

X_test  = df_test[FEATURES].values
y_test  = df_test[TARGETS].values

log(f"X_train shape : {X_train.shape}")
log(f"y_train shape : {y_train.shape}")
log(f"Features      : {FEATURES}")
log(f"Targets       : {TARGETS}")

# ── Scale features ──
# XGBoost is tree-based so doesn't strictly need scaling,
# but scaling helps with regularisation and interpretation
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

log(f"Features scaled (StandardScaler fitted on train only)")

# ══════════════════════════════════════════════
#  STEP 3 — TRAIN MODEL
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 3 — Train XGBoost Controller Model")
print(SEP)

log(f"XGBoost parameters:")
for k, v in XGB_PARAMS.items():
    log(f"  {k:20s} = {v}")

# MultiOutputRegressor trains one XGBoost tree per target
# u_d and u_q are predicted independently
base_model = XGBRegressor(**XGB_PARAMS)
model = MultiOutputRegressor(base_model, n_jobs=1)

log(f"\nTraining started...")
t_start = time.time()
model.fit(X_train_sc, y_train)
t_elapsed = time.time() - t_start

log(f"Training complete in {t_elapsed:.1f} seconds")

# ══════════════════════════════════════════════
#  STEP 4 — EVALUATE ON VALIDATION SET
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 4 — Evaluate on validation set")
print(SEP)

y_val_pred = model.predict(X_val_sc)

log(f"{'Target':<20} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
log(f"{'-'*52}")

val_metrics = {}
for i, target in enumerate(TARGETS):
    rmse = np.sqrt(mean_squared_error(y_val[:, i], y_val_pred[:, i]))
    mae  = mean_absolute_error(y_val[:, i], y_val_pred[:, i])
    r2   = r2_score(y_val[:, i], y_val_pred[:, i])
    val_metrics[target] = {'rmse': rmse, 'mae': mae, 'r2': r2}
    log(f"  {target:<18} {rmse:>10.4f} {mae:>10.4f} {r2:>10.4f}")

# ══════════════════════════════════════════════
#  STEP 5 — EVALUATE ON TEST SET
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 5 — Evaluate on test set (final)")
print(SEP)

y_test_pred = model.predict(X_test_sc)

log(f"{'Target':<20} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
log(f"{'-'*52}")

test_metrics = {}
for i, target in enumerate(TARGETS):
    rmse = np.sqrt(mean_squared_error(y_test[:, i], y_test_pred[:, i]))
    mae  = mean_absolute_error(y_test[:, i], y_test_pred[:, i])
    r2   = r2_score(y_test[:, i], y_test_pred[:, i])
    test_metrics[target] = {'rmse': rmse, 'mae': mae, 'r2': r2}
    log(f"  {target:<18} {rmse:>10.4f} {mae:>10.4f} {r2:>10.4f}")

# ── Interpretation guide ──
print()
log("R² Score Guide:")
log("  R² > 0.95 → Excellent  ✅")
log("  R² > 0.90 → Good       ✅")
log("  R² > 0.80 → Acceptable ⚠️")
log("  R² < 0.80 → Needs work ❌")

# ══════════════════════════════════════════════
#  STEP 6 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 6 — Feature importance")
print(SEP)

for i, target in enumerate(TARGETS):
    importance = model.estimators_[i].feature_importances_
    fi_df = pd.DataFrame({
        'feature'   : FEATURES,
        'importance': importance
    }).sort_values('importance', ascending=False)
    log(f"\nTop features for predicting {target}:")
    for _, row in fi_df.iterrows():
        bar = '█' * int(row['importance'] * 40)
        log(f"  {row['feature']:<25} {row['importance']:.4f}  {bar}")

# ══════════════════════════════════════════════
#  STEP 7 — SAVE PLOTS
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 7 — Save diagnostic plots")
print(SEP)

os.makedirs(PLOTS_DIR, exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Motor Controller Model — Predictions vs Actual', fontsize=14)

# Sample 5000 points for plotting (plotting all 200k is too slow)
n_plot = min(5000, len(y_test))
idx    = np.random.choice(len(y_test), n_plot, replace=False)

for i, target in enumerate(TARGETS):
    # Scatter: predicted vs actual
    ax = axes[i, 0]
    ax.scatter(y_test[idx, i], y_test_pred[idx, i],
               alpha=0.3, s=5, color='steelblue')
    mn = min(y_test[:, i].min(), y_test_pred[:, i].min())
    mx = max(y_test[:, i].max(), y_test_pred[:, i].max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect fit')
    ax.set_xlabel(f'Actual {target}')
    ax.set_ylabel(f'Predicted {target}')
    r2 = test_metrics[target]['r2']
    ax.set_title(f'{target} — R²={r2:.4f}')
    ax.legend()

    # Residuals over time
    ax2 = axes[i, 1]
    residuals = y_test[:, i] - y_test_pred[:, i]
    ax2.plot(residuals[:2000], linewidth=0.5, color='darkorange', alpha=0.7)
    ax2.axhline(0, color='red', linestyle='--', linewidth=1)
    ax2.set_xlabel('Sample index')
    ax2.set_ylabel('Residual (Actual − Predicted)')
    rmse = test_metrics[target]['rmse']
    ax2.set_title(f'{target} residuals — RMSE={rmse:.4f}')

plt.tight_layout()
plot_path = os.path.join(PLOTS_DIR, 'controller_model_diagnostics.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
log(f"Saved: {plot_path}")

# Feature importance plot
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Controller Model — Feature Importance', fontsize=13)

for i, target in enumerate(TARGETS):
    importance = model.estimators_[i].feature_importances_
    fi_df = pd.DataFrame({'feature': FEATURES, 'importance': importance})
    fi_df = fi_df.sort_values('importance', ascending=True)
    axes2[i].barh(fi_df['feature'], fi_df['importance'], color='steelblue')
    axes2[i].set_title(f'Feature Importance — {target}')
    axes2[i].set_xlabel('Importance Score')

plt.tight_layout()
fi_path = os.path.join(PLOTS_DIR, 'controller_feature_importance.png')
plt.savefig(fi_path, dpi=150, bbox_inches='tight')
plt.close()
log(f"Saved: {fi_path}")

# ══════════════════════════════════════════════
#  STEP 8 — SAVE MODEL AND SCALER
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 8 — Save model and scaler")
print(SEP)

os.makedirs(MODEL_DIR, exist_ok=True)

model_path  = os.path.join(MODEL_DIR, "controller_model.joblib")
scaler_path = os.path.join(MODEL_DIR, "controller_scaler.joblib")

joblib.dump(model,  model_path)
joblib.dump(scaler, scaler_path)

log(f"Saved model  : {model_path}")
log(f"Saved scaler : {scaler_path}")

# ── Save metrics to file for reference ──
metrics_path = os.path.join(MODEL_DIR, "controller_metrics.txt")
with open(metrics_path, 'w') as f:
    f.write("Motor Controller Model — Evaluation Metrics\n")
    f.write("=" * 50 + "\n\n")
    f.write("VALIDATION SET\n")
    for target, m in val_metrics.items():
        f.write(f"  {target}: RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}, R²={m['r2']:.4f}\n")
    f.write("\nTEST SET\n")
    for target, m in test_metrics.items():
        f.write(f"  {target}: RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}, R²={m['r2']:.4f}\n")
    f.write(f"\nTraining time: {t_elapsed:.1f} seconds\n")
    f.write(f"Training rows: {len(df_train):,}\n")
    f.write(f"Features used: {FEATURES}\n")

log(f"Saved metrics: {metrics_path}")

# ══════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  CONTROLLER MODEL COMPLETE")
print(SEP)
print(f"""
  Model     : XGBoost MultiOutputRegressor
  Features  : {len(FEATURES)}  →  {FEATURES}
  Targets   : {TARGETS}
  Train rows: {len(df_train):,}
  Train time: {t_elapsed:.1f} seconds

  TEST RESULTS:
""")
for target, m in test_metrics.items():
    status = "✅" if m['r2'] > 0.90 else ("⚠️" if m['r2'] > 0.80 else "❌")
    print(f"    {target:<6} R²={m['r2']:.4f}  RMSE={m['rmse']:.4f}  {status}")

print(f"""
  Saved to {MODEL_DIR}/
    controller_model.joblib   ← load this for Digital Twin
    controller_scaler.joblib  ← apply before prediction
    controller_metrics.txt    ← full metrics report

  Next step:
    Run model_motor.py  (LSTM Motor Performance model)
    — install tensorflow first if not done:
      pip install tensorflow
""")
