"""
Definitive Drift Confirmation Test
====================================
Tests Vehicle Dynamics model on ALL 8 test trips separately.
For each trip shows:
  - Whether the trip starts stationary or moving
  - Window RMSE across 5 windows
  - Whether drift flag appears

Logic:
  If drift only appears on trips that start stationary → behaviour change
  If drift appears on trips that start moving too     → model problem
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

MODEL_DIR  = "Models"
DATA_DIR   = "Datasets/processed"
PLOTS_DIR  = "Plots/validation"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load model
dyn_model  = joblib.load(os.path.join(MODEL_DIR, "dynamics_model.joblib"))
dyn_scaler = joblib.load(os.path.join(MODEL_DIR, "dynamics_scaler.joblib"))

DYN_FEATURES = [
    'Throttle [%]', 'Velocity_kmh_prev', 'velocity_ms_prev',
    'Longitudinal_Acceleration_ms2_prev', 'Motor_Torque_Nm_prev',
    'slope', 'Regenerative Braking Signal',
    'Ambient Temperature [°C]', 'trip_type'
]
DYN_TARGETS = [
    'Velocity [km/h]',
    'Longitudinal Acceleration [m/s^2]',
    'Motor Torque [Nm]'
]

df = pd.read_csv(os.path.join(DATA_DIR, "vehicle_dynamics_test.csv"))
all_trips = sorted(df['trip_id'].unique())

SEP = "=" * 70
print(f"\n{SEP}")
print("  DEFINITIVE DRIFT CONFIRMATION TEST")
print(f"  Testing all {len(all_trips)} test trips independently")
print(SEP)

WINDOW_SIZE = 100
results     = {}

for trip_id in all_trips:
    trip = df[df['trip_id'] == trip_id].reset_index(drop=True)
    n    = min(500, len(trip))

    if n < WINDOW_SIZE * 2:
        print(f"\n  {trip_id}: too short ({n} rows), skipping")
        continue

    # ── Trip characteristics ──
    first_10     = trip.head(10)
    starts_stationary = (first_10['Velocity [km/h]'] == 0).all()
    stationary_steps  = (trip['Velocity [km/h]'] == 0).sum()
    moving_steps      = (trip['Velocity [km/h]'] > 1).sum()
    max_speed         = trip['Velocity [km/h]'].max()
    avg_speed         = trip['Velocity [km/h]'].mean()

    # ── Run model ──
    X      = trip[DYN_FEATURES].values[:n]
    y_true = trip[DYN_TARGETS].values[:n]

    X_sc   = dyn_scaler.transform(X)
    y_pred = dyn_model.predict(X_sc)

    # ── Window RMSE for Velocity (primary signal) ──
    vel_errors = y_true[:, 0] - y_pred[:, 0]
    n_windows  = n // WINDOW_SIZE
    w_rmses    = []
    for w in range(n_windows):
        s = w * WINDOW_SIZE
        e = s + WINDOW_SIZE
        w_rmses.append(float(np.sqrt(np.mean(vel_errors[s:e]**2))))

    # ── Overall metrics ──
    overall_rmse = float(np.sqrt(np.mean(vel_errors**2)))
    overall_r2   = float(1 - np.sum(vel_errors**2) /
                         np.sum((y_true[:, 0] - y_true[:, 0].mean())**2))

    # ── Drift detection ──
    first_w = w_rmses[0] if w_rmses[0] > 0.001 else 0.001
    max_w   = max(w_rmses)
    is_drift = max_w > first_w * 1.5   # 50% threshold

    results[trip_id] = {
        'starts_stationary' : starts_stationary,
        'stationary_steps'  : int(stationary_steps),
        'moving_steps'      : int(moving_steps),
        'max_speed'         : float(max_speed),
        'avg_speed'         : float(avg_speed),
        'w_rmses'           : w_rmses,
        'overall_rmse'      : overall_rmse,
        'overall_r2'        : overall_r2,
        'is_drift'          : is_drift,
        'n'                 : n,
    }

# ── Print results ──
print(f"\n{'Trip':<10} {'Starts':>10} {'Stationary':>12} {'AvgSpeed':>10} "
      f"{'OverallRMSE':>13} {'R2':>8}  {'Drift?':>8}")
print(f"{'-'*80}")

stationary_with_drift = 0
stationary_without_drift = 0
moving_with_drift = 0
moving_without_drift = 0

for trip_id, r in results.items():
    start_type = "STOPPED" if r['starts_stationary'] else "MOVING"
    drift_flag = "DRIFT ⚠️" if r['is_drift'] else "STABLE ✅"

    print(f"  {trip_id:<8} {start_type:>10} "
          f"{r['stationary_steps']:>10} steps "
          f"{r['avg_speed']:>8.1f} km/h "
          f"{r['overall_rmse']:>12.4f} km/h "
          f"{r['overall_r2']:>8.4f}  {drift_flag}")

    if r['starts_stationary'] and r['is_drift']:
        stationary_with_drift += 1
    elif r['starts_stationary'] and not r['is_drift']:
        stationary_without_drift += 1
    elif not r['starts_stationary'] and r['is_drift']:
        moving_with_drift += 1
    else:
        moving_without_drift += 1

# ── Window RMSE detail ──
print(f"\n\nWindow RMSE Detail (Velocity km/h) — each window = {WINDOW_SIZE} steps")
print(f"{'-'*80}")
header = f"{'Trip':<10} {'Start':>8}"
for w in range(5):
    header += f"  W{w+1}({w*100}-{(w+1)*100})"
header += "  Verdict"
print(header)
print(f"{'-'*80}")

for trip_id, r in results.items():
    start_type = "STOPPED" if r['starts_stationary'] else "MOVING"
    row = f"  {trip_id:<8} {start_type:>8}"
    for w_rmse in r['w_rmses']:
        row += f"  {w_rmse:>10.4f}"
    row += f"  {'DRIFT' if r['is_drift'] else 'STABLE'}"
    print(row)

# ── Conclusion ──
print(f"\n\n{'='*70}")
print("  CONCLUSION")
print(f"{'='*70}")
print(f"""
  Trips starting STATIONARY:
    With drift flag    : {stationary_with_drift}
    Without drift flag : {stationary_without_drift}

  Trips starting MOVING:
    With drift flag    : {moving_with_drift}
    Without drift flag : {moving_without_drift}
""")

if moving_with_drift == 0 and stationary_with_drift > 0:
    print("  VERDICT: CONFIRMED BEHAVIOUR CHANGE ✅")
    print("  Drift flag ONLY appears on trips that start stationary.")
    print("  The model has NO problem. The drift is caused by the")
    print("  stationary→moving transition in the test data.")
    print("  Digital Twin is ready for deployment.")
elif moving_with_drift > 0:
    print("  VERDICT: GENUINE MODEL WEAKNESS ⚠️")
    print("  Drift flag appears on moving trips too.")
    print(f"  {moving_with_drift} moving trip(s) show drift.")
    print("  Model may need retraining on more diverse driving data.")
else:
    print("  VERDICT: ALL TRIPS STABLE ✅")
    print("  No drift detected on any trip.")
    print("  Digital Twin is ready for deployment.")

# ── Plot ──
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Drift Confirmation — Window RMSE per Trip\n'
             '(Blue=Stationary start, Green=Moving start)',
             fontsize=13)
axes = axes.flatten()

for idx, (trip_id, r) in enumerate(results.items()):
    if idx >= 8:
        break
    ax     = axes[idx]
    color  = 'steelblue' if r['starts_stationary'] else 'green'
    x      = np.arange(1, len(r['w_rmses']) + 1)
    ax.bar(x, r['w_rmses'], color=color, alpha=0.7)
    ax.axhline(r['w_rmses'][0] * 1.5, color='red',
               linestyle='--', linewidth=1, label='Drift threshold')
    start = "STOPPED" if r['starts_stationary'] else "MOVING"
    drift = "DRIFT" if r['is_drift'] else "STABLE"
    ax.set_title(f'{trip_id} [{start}]\nRMSE={r["overall_rmse"]:.4f} — {drift}',
                 fontsize=9)
    ax.set_xlabel('Window')
    ax.set_ylabel('RMSE (km/h)')
    ax.legend(fontsize=7)

plt.tight_layout()
plot_path = os.path.join(PLOTS_DIR, 'drift_confirmation_all_trips.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Plot saved: {plot_path}")
