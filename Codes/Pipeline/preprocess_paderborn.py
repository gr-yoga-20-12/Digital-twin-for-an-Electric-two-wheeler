"""
Paderborn PMSM — Preprocessing Script
=======================================
Dataset : measures_v2.csv  (1,330,816 rows, 69 sessions)
Sampling: 2 Hz  →  Δt = 0.5 seconds per row
Output  : processed_motor_data.csv

What this script does
─────────────────────
1. Derives missing metrics
     • Throttle proxy    — from i_q (FOC physics)
     • Duty cycle        — from u_d, u_q magnitude
     • Load torque       — from Newton's law on motor shaft
2. Creates lagged features (t-1 state)
     — applied WITHIN each profile_id session (no boundary leakage)
3. Splits sessions into Train / Validation / Test
     — split by session ID, not by row (prevents data leakage)
4. Exports three CSVs ready for model training
"""

import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
INPUT_FILE  = "Datasets/archive/measures_v2.csv"        # Paderborn CSV
OUTPUT_DIR  = "Datasets/processed"              # output folder
DT          = 0.5                      # seconds per row (2 Hz)
VBUS        = 48.0                     # nominal DC bus voltage (V)
J           = 0.0011                   # rotor inertia  (kg·m²)
B           = 0.0015                   # friction coeff (Nm·s/rad)

# Train / Val / Test split by session count
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15  (remainder)

# ─────────────────────────────────────────────
SEP = "=" * 55

def log(msg): print(f"  {msg}")

# ══════════════════════════════════════════════
#  LOAD
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 1 — Load dataset")
print(SEP)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Cannot find '{INPUT_FILE}'.\n"
        f"Make sure measures_v2.csv is in the same folder as this script."
    )

df = pd.read_csv(INPUT_FILE)
log(f"Loaded  : {len(df):,} rows  ×  {len(df.columns)} columns")
log(f"Sessions: {df['profile_id'].nunique()}  (profile_id)")
log(f"Columns : {list(df.columns)}")

# ══════════════════════════════════════════════
#  STEP 2 — DERIVE MISSING METRICS
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 2 — Derive missing metrics")
print(SEP)

# ── 2a. Throttle proxy ──
# In FOC control, i_q is the torque-producing current.
# Throttle directly commands i_q reference.
# Normalise to 0-100 range using max observed i_q.
# Negative i_q = regenerative braking (valid, preserve sign).
IQ_MAX = df['i_q'].abs().max()
df['throttle_proxy'] = (df['i_q'] / IQ_MAX) * 100
log(f"throttle_proxy : range [{df['throttle_proxy'].min():.1f}, "
    f"{df['throttle_proxy'].max():.1f}]  (negative = regen braking)")

# ── 2b. Duty cycle ──
# D = |V_phase| / Vbus
# V_phase magnitude from d-q voltage components
df['v_phase_mag'] = np.sqrt(df['u_d']**2 + df['u_q']**2)
df['duty_cycle']  = df['v_phase_mag'] / VBUS
df['duty_cycle']  = df['duty_cycle'].clip(0, 1)   # physical bounds
log(f"duty_cycle     : range [{df['duty_cycle'].min():.4f}, "
    f"{df['duty_cycle'].max():.4f}]")

# ── 2c. Load torque ──
# T_load = T_electromagnetic - J*(dω/dt) - B*ω
# dω/dt computed WITHIN each session (no cross-session diff)
# motor_speed is in RPM — convert to rad/s for physics
df['omega_rad'] = df['motor_speed'] * (2 * np.pi / 60)  # RPM → rad/s

# Compute dω/dt within each session
df['domega_dt'] = (
    df.groupby('profile_id')['omega_rad']
    .transform(lambda x: x.diff() / DT)
)

df['t_load'] = (
    df['torque']
    - (J * df['domega_dt'].fillna(0))
    - (B * df['omega_rad'])
)
log(f"t_load         : range [{df['t_load'].min():.2f}, "
    f"{df['t_load'].max():.2f}]  Nm")

# ── 2d. Electrical power ──
# P = u_d*i_d + u_q*i_q  (d-q power formula)
df['power_elec'] = df['u_d'] * df['i_d'] + df['u_q'] * df['i_q']
log(f"power_elec     : range [{df['power_elec'].min():.1f}, "
    f"{df['power_elec'].max():.1f}]  W")

# ── 2e. Motor efficiency proxy ──
# η = P_mechanical / P_electrical
# P_mech = torque × ω_rad
df['power_mech'] = df['torque'] * df['omega_rad']
# Avoid division by zero; clip efficiency to physical range
with np.errstate(divide='ignore', invalid='ignore'):
    df['efficiency'] = np.where(
        df['power_elec'].abs() > 1.0,
        (df['power_mech'] / df['power_elec']).clip(-1, 1),
        0.0
    )
log(f"efficiency     : range [{df['efficiency'].min():.3f}, "
    f"{df['efficiency'].max():.3f}]")

# ══════════════════════════════════════════════
#  STEP 3 — LAGGED FEATURES (t-1 state)
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 3 — Create lagged features (within sessions)")
print(SEP)

# These are the feedback loop inputs:
# controller receives motor state from previous timestep
LAG_COLS = [
    'i_d', 'i_q',
    'motor_speed', 'omega_rad',
    'pm',               # rotor temp  — primary thermal target
    'stator_winding',   # stator temp — thermal load indicator
    'stator_tooth',
    'stator_yoke',
    'torque',
    'duty_cycle',
    'throttle_proxy',
]

for col in LAG_COLS:
    if col in df.columns:
        # shift(1) within each session — no boundary leakage
        df[f'{col}_prev'] = df.groupby('profile_id')[col].shift(1)

log(f"Lagged columns created: {[c+'_prev' for c in LAG_COLS if c in df.columns]}")

# ── Drop first row of each session (NaN from shift) ──
rows_before = len(df)
df.dropna(inplace=True)
log(f"Dropped {rows_before - len(df):,} boundary rows  "
    f"({len(df):,} rows remain)")

# ══════════════════════════════════════════════
#  STEP 4 — SANITY CHECKS
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 4 — Sanity checks")
print(SEP)

# Physical constraint checks
checks = {
    "throttle_proxy in [-100, 100]":
        df['throttle_proxy'].between(-100, 100).all(),
    "duty_cycle in [0, 1]":
        df['duty_cycle'].between(0, 1).all(),
    "pm (rotor temp) > 0°C":
        (df['pm'] > 0).all(),
    "stator_winding > 0°C":
        (df['stator_winding'] > 0).all(),
    "No NaN in key columns":
        df[['i_d','i_q','motor_speed','torque','pm',
            'u_d','u_q','ambient']].isnull().sum().sum() == 0,
}

all_pass = True
for check, result in checks.items():
    status = "✅ PASS" if result else "❌ FAIL"
    log(f"{status}  —  {check}")
    if not result:
        all_pass = False

if all_pass:
    log("All sanity checks passed.")
else:
    log("WARNING: Some checks failed — review before training.")

# ══════════════════════════════════════════════
#  STEP 5 — TRAIN / VAL / TEST SPLIT
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 5 — Train / Validation / Test split")
print(SEP)

# Split by session ID — never by row
# This prevents the model from seeing future timesteps during training
all_pids   = sorted(df['profile_id'].unique())
n          = len(all_pids)
n_train    = int(n * TRAIN_RATIO)
n_val      = int(n * VAL_RATIO)

train_pids = all_pids[:n_train]
val_pids   = all_pids[n_train:n_train + n_val]
test_pids  = all_pids[n_train + n_val:]

df_train = df[df['profile_id'].isin(train_pids)].copy()
df_val   = df[df['profile_id'].isin(val_pids)].copy()
df_test  = df[df['profile_id'].isin(test_pids)].copy()

log(f"Train sessions : {len(train_pids)}  →  {len(df_train):,} rows")
log(f"Val   sessions : {len(val_pids)}  →  {len(df_val):,} rows")
log(f"Test  sessions : {len(test_pids)}  →  {len(df_test):,} rows")
log(f"Train pids : {train_pids}")
log(f"Val   pids : {val_pids}")
log(f"Test  pids : {test_pids}")

# ══════════════════════════════════════════════
#  STEP 6 — DEFINE MODEL FEATURE SETS
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 6 — Feature sets for each model")
print(SEP)

# ── Controller Model (XGBoost) ──
# Input:  throttle proxy + feedback from motor (t-1)
# Output: u_d, u_q  (phase voltages)
CONTROLLER_FEATURES = [
    'throttle_proxy',       # rider intent (derived from i_q)
    'i_d_prev',             # previous d-axis current
    'i_q_prev',             # previous q-axis current
    'motor_speed_prev',     # previous RPM
    'pm_prev',              # previous rotor temperature
    'stator_winding_prev',  # previous stator temperature
    'ambient',              # ambient temperature
    'duty_cycle_prev',      # previous duty cycle
]
CONTROLLER_TARGETS = ['u_d', 'u_q']

# ── Motor Performance Model (LSTM) ──
# Input:  controller output (u_d, u_q) + motor state (t-1)
# Output: i_d, i_q, torque, motor_speed, pm (rotor temp)
MOTOR_FEATURES = [
    'u_d',                  # d-axis voltage  ← from controller
    'u_q',                  # q-axis voltage  ← from controller
    'i_d_prev',             # previous d-axis current
    'i_q_prev',             # previous q-axis current
    'motor_speed_prev',     # previous RPM
    'omega_rad',            # current speed in rad/s (for physics)
    'pm_prev',              # previous rotor temperature
    'stator_winding_prev',  # previous stator temp (thermal load)
    'stator_tooth_prev',    # previous stator tooth temp
    'stator_yoke_prev',     # previous stator yoke temp
    'ambient',              # ambient temperature
    't_load',               # load torque (derived)
]
MOTOR_TARGETS = ['i_d', 'i_q', 'torque', 'motor_speed', 'pm',
                 'stator_winding']

log("Controller model")
log(f"  Features : {CONTROLLER_FEATURES}")
log(f"  Targets  : {CONTROLLER_TARGETS}")
log("")
log("Motor performance model")
log(f"  Features : {MOTOR_FEATURES}")
log(f"  Targets  : {MOTOR_TARGETS}")

# ══════════════════════════════════════════════
#  STEP 7 — EXPORT
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 7 — Export processed CSVs")
print(SEP)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Full processed dataset (all sessions)
full_path = os.path.join(OUTPUT_DIR, "processed_motor_data.csv")
df.to_csv(full_path, index=False)
log(f"Saved full    : {full_path}  ({len(df):,} rows)")

# Split files
train_path = os.path.join(OUTPUT_DIR, "motor_train.csv")
val_path   = os.path.join(OUTPUT_DIR, "motor_val.csv")
test_path  = os.path.join(OUTPUT_DIR, "motor_test.csv")

df_train.to_csv(train_path, index=False)
df_val.to_csv(val_path,     index=False)
df_test.to_csv(test_path,   index=False)

log(f"Saved train   : {train_path}  ({len(df_train):,} rows)")
log(f"Saved val     : {val_path}  ({len(df_val):,} rows)")
log(f"Saved test    : {test_path}  ({len(df_test):,} rows)")

# ── Final summary ──
print(f"\n{SEP}")
print("  PREPROCESSING COMPLETE")
print(SEP)
print(f"""
  Input  : {INPUT_FILE}  ({len(pd.read_csv(INPUT_FILE)):,} original rows)
  Output : {OUTPUT_DIR}/
             ├── processed_motor_data.csv  (full)
             ├── motor_train.csv
             ├── motor_val.csv
             └── motor_test.csv

  Derived columns added:
    throttle_proxy  — rider intent via i_q / i_q_max × 100
    duty_cycle      — |V_phase| / Vbus
    t_load          — Newton's law on motor shaft
    power_elec      — d-q electrical power
    power_mech      — torque × ω
    efficiency      — power_mech / power_elec
    omega_rad       — motor_speed in rad/s
    domega_dt       — dω/dt within each session
    *_prev columns  — lagged t-1 state (within sessions only)

  Next step:
    Run preprocess_paderborn.py → check output →
    then run model_controller.py + model_motor.py
""")
