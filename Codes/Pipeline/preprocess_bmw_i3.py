"""
BMW i3 Real Driving Cycles — Preprocessing Script
===================================================
Dataset  : TripA01–TripA32  (23 columns, summer)
           TripB01–TripB38  (48 columns, winter)
           Overview.xlsx    (trip-level metadata)
Sampling : 10 Hz  →  Δt = 0.1 seconds per row
Total    : 70 trips  ≈  1.8 million rows

What this script does
──────────────────────
1. Loads all TripA + TripB CSV files from your dataset folder
2. Derives features needed for both models:
     • Slope          — from elevation change within trip
     • Distance step  — from velocity × Δt
     • Energy step    — from voltage × current × Δt
     • Energy normalised by mass (Wh/kg) — removes car vs scooter size diff
     • Power          — voltage × current (W)
     • Lagged features (t-1 state) — within each trip only
3. Selects the correct columns for each model
4. Splits trips into Train / Validation / Test (by trip ID, not by row)
5. Exports clean CSVs ready for model training

Output files
─────────────
processed/
  ├── vehicle_dynamics_train.csv
  ├── vehicle_dynamics_val.csv
  ├── vehicle_dynamics_test.csv
  ├── range_management_train.csv
  ├── range_management_val.csv
  └── range_management_test.csv
"""

import pandas as pd
import numpy as np
import os
import glob

# ─────────────────────────────────────────────
#  CONFIGURATION — Update this path to your folder
# ─────────────────────────────────────────────

# Folder containing all TripA*.csv and TripB*.csv files
TRIP_FOLDER = "Datasets/raw/BMW_i3"

# Output folder
OUTPUT_DIR  = "Datasets/processed"

# Sampling interval (10 Hz)
DT = 0.1  # seconds

# Vehicle masses for energy normalisation
BMW_MASS     = 1270.0   # kg  (BMW i3 60Ah kerb weight)
SCOOTER_MASS =   90.0   # kg  (scooter 15kg + rider 75kg)

# Train / Val / Test split ratios (by trip count)
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
# TEST_RATIO  = 0.10  (remainder)

# ─────────────────────────────────────────────
SEP = "=" * 60

def log(msg): print(f"  {msg}")


# ══════════════════════════════════════════════
#  STEP 1 — FIND AND LOAD ALL TRIP FILES
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 1 — Load all trip files")
print(SEP)

# Find all TripA and TripB CSV files
tripA_files = sorted(glob.glob(os.path.join(TRIP_FOLDER, "TripA*.csv")))
tripB_files = sorted(glob.glob(os.path.join(TRIP_FOLDER, "TripB*.csv")))
all_files   = tripA_files + tripB_files

if len(all_files) == 0:
    raise FileNotFoundError(
        f"\nNo trip CSV files found in '{TRIP_FOLDER}'.\n"
        f"Make sure TripA*.csv and TripB*.csv files are in that folder.\n"
        f"Update TRIP_FOLDER at the top of this script if needed."
    )

log(f"Found TripA files : {len(tripA_files)}")
log(f"Found TripB files : {len(tripB_files)}")
log(f"Total trip files  : {len(all_files)}")

# ── Core columns needed from BOTH TripA and TripB ──
# These exist in both 23-column and 48-column files
CORE_COLS = [
    'Time [s]',
    'Velocity [km/h]',
    'Elevation [m]',
    'Throttle [%]',
    'Motor Torque [Nm]',
    'Longitudinal Acceleration [m/s^2]',
    'Regenerative Braking Signal ',       # note trailing space — matches raw file
    'Battery Voltage [V]',
    'Battery Current [A]',
    'Battery Temperature [°C]',
    'SoC [%]',
    'Ambient Temperature [°C]',
]

all_trips = []
skipped   = []

for fpath in all_files:
    trip_id = os.path.basename(fpath).replace('.csv', '')
    try:
        df = pd.read_csv(fpath, encoding='latin-1', sep=None, engine='python')

        # Rename trailing-space column if present
        df.columns = [c.rstrip() if c.strip() == 'Regenerative Braking Signal'
                      else c for c in df.columns]

        # Keep only core columns
        missing = [c for c in CORE_COLS if c not in df.columns]
        if missing:
            # Try stripping trailing spaces from all column names
            df.columns = [c.strip() for c in df.columns]
            CORE_COLS_STRIPPED = [c.strip() for c in CORE_COLS]
            missing = [c for c in CORE_COLS_STRIPPED if c not in df.columns]
            if missing:
                log(f"  SKIP {trip_id} — missing columns: {missing}")
                skipped.append(trip_id)
                continue
            df = df[CORE_COLS_STRIPPED].copy()
            df.columns = CORE_COLS  # restore original names
        else:
            df = df[CORE_COLS].copy()

        # Tag each row with trip identity
        df['trip_id']   = trip_id
        df['trip_type'] = 'A' if 'TripA' in trip_id else 'B'

        all_trips.append(df)

    except Exception as e:
        log(f"  SKIP {trip_id} — error: {e}")
        skipped.append(trip_id)

log(f"\nLoaded successfully : {len(all_trips)} trips")
if skipped:
    log(f"Skipped            : {skipped}")

# Combine all trips into one dataframe
df = pd.concat(all_trips, ignore_index=True)

# Standardise column names (strip spaces)
df.columns = [c.strip() for c in df.columns]

log(f"Combined rows      : {len(df):,}")
log(f"Columns            : {list(df.columns)}")


# ══════════════════════════════════════════════
#  STEP 2 — CLEAN AND VALIDATE RAW DATA
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 2 — Clean and validate raw data")
print(SEP)

# ── 2a. Drop rows with nulls in core columns ──
rows_before = len(df)
df.dropna(subset=[c.strip() for c in CORE_COLS], inplace=True)
log(f"Dropped null rows  : {rows_before - len(df)}")

# ── 2b. Physical range checks — clip outliers ──
# Velocity: clip negative (GPS noise) and extreme values
df['Velocity [km/h]'] = df['Velocity [km/h]'].clip(lower=0, upper=200)

# SoC: must be 0–100
df['SoC [%]'] = df['SoC [%]'].clip(lower=0, upper=100)

# Throttle: must be 0–100
df['Throttle [%]'] = df['Throttle [%]'].clip(lower=0, upper=100)

# Battery current: negative = discharging in BMW i3 convention
# Keep as-is — the sign is meaningful for energy calculation

log(f"Physical range clips applied (velocity, SoC, throttle)")
log(f"Rows after cleaning: {len(df):,}")


# ══════════════════════════════════════════════
#  STEP 3 — DERIVE FEATURES
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 3 — Derive features (within each trip)")
print(SEP)

# All derived features use groupby(trip_id) to prevent
# cross-trip boundary contamination

# ── 3a. Velocity in m/s ──
df['velocity_ms'] = df['Velocity [km/h]'] / 3.6

# ── 3b. Distance step (metres per timestep) ──
df['distance_step_m'] = df['velocity_ms'] * DT

# ── 3c. Slope (rise/run, dimensionless) ──
# Derived within each trip from elevation change
df['delta_elevation'] = (
    df.groupby('trip_id')['Elevation [m]']
    .transform(lambda x: x.diff().fillna(0))
)
# Avoid division by zero at standstill
with np.errstate(divide='ignore', invalid='ignore'):
    df['slope'] = np.where(
        df['distance_step_m'] > 0.01,
        df['delta_elevation'] / df['distance_step_m'],
        0.0
    )
# Clip slope to realistic road range (-30% to +30% grade)
df['slope'] = df['slope'].clip(-0.30, 0.30)
log(f"slope              : range [{df['slope'].min():.4f}, {df['slope'].max():.4f}]")

# ── 3d. Electrical power (W) ──
# BMW i3: positive current = charging, negative = discharging
# Power drawn from battery = -V × I  (positive when driving)
df['power_W'] = df['Battery Voltage [V]'] * df['Battery Current [A]'] * -1

# ── 3e. Energy per step (Wh) ──
df['energy_step_Wh'] = df['power_W'] * DT / 3600

# ── 3f. Mass-normalised energy (Wh/kg) ──
# KEY STEP: dividing by BMW mass removes vehicle-size dependency
# The model learns Wh/kg which is universal across EV types
df['energy_norm_Whkg'] = df['energy_step_Wh'] / BMW_MASS

# ── 3g. Cumulative energy per trip (Wh) ──
df['energy_cumulative_Wh'] = (
    df.groupby('trip_id')['energy_step_Wh']
    .transform('cumsum')
)

# ── 3h. SoC change per step ──
df['delta_soc'] = (
    df.groupby('trip_id')['SoC [%]']
    .transform(lambda x: x.diff().fillna(0))
)

# ── 3i. Jerk (rate of acceleration change) ──
df['jerk'] = (
    df.groupby('trip_id')['Longitudinal Acceleration [m/s^2]']
    .transform(lambda x: x.diff().fillna(0) / DT)
).clip(-20, 20)

log(f"energy_norm_Whkg   : range [{df['energy_norm_Whkg'].min():.6f}, {df['energy_norm_Whkg'].max():.6f}]")
log(f"delta_soc          : range [{df['delta_soc'].min():.4f}, {df['delta_soc'].max():.4f}]")
log(f"power_W            : range [{df['power_W'].min():.1f}, {df['power_W'].max():.1f}]")


# ══════════════════════════════════════════════
#  STEP 4 — LAGGED FEATURES (t-1 state)
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 4 — Create lagged features (within trips)")
print(SEP)

LAG_COLS = [
    'Velocity [km/h]',
    'velocity_ms',
    'Longitudinal Acceleration [m/s^2]',
    'Motor Torque [Nm]',
    'SoC [%]',
    'Throttle [%]',
    'energy_norm_Whkg',
    'slope',
    'power_W',
]

for col in LAG_COLS:
    safe_name = col.replace('[', '').replace(']', '').replace('/', '').replace('^', '').replace(' ', '_').replace('%', 'pct').replace('°', '')
    df[f'{safe_name}_prev'] = (
        df.groupby('trip_id')[col]
        .transform(lambda x: x.shift(1))
    )

# Drop first row of each trip (NaN from shift)
rows_before = len(df)
df.dropna(inplace=True)
log(f"Dropped boundary rows : {rows_before - len(df)} (one per trip)")
log(f"Rows remaining        : {len(df):,}")


# ══════════════════════════════════════════════
#  STEP 5 — SANITY CHECKS
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 5 — Sanity checks")
print(SEP)

checks = {
    "Throttle in [0, 100]":
        df['Throttle [%]'].between(0, 100).all(),
    "SoC in [0, 100]":
        df['SoC [%]'].between(0, 100).all(),
    "Velocity >= 0":
        (df['Velocity [km/h]'] >= 0).all(),
    "Slope in [-0.30, 0.30]":
        df['slope'].between(-0.30, 0.30).all(),
    "No NaN in key columns":
        df[['Velocity [km/h]', 'Throttle [%]', 'SoC [%]',
            'Motor Torque [Nm]', 'energy_norm_Whkg']].isnull().sum().sum() == 0,
    "Both trip types present":
        df['trip_type'].nunique() == 2,
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
#  STEP 6 — TRAIN / VAL / TEST SPLIT BY TRIP
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 6 — Train / Validation / Test split (by trip ID)")
print(SEP)

# Split by trip ID — NEVER by row
# This prevents the model seeing future timesteps during training
all_trip_ids = sorted(df['trip_id'].unique())
n            = len(all_trip_ids)
n_train      = int(n * TRAIN_RATIO)
n_val        = int(n * VAL_RATIO)

train_trips  = all_trip_ids[:n_train]
val_trips    = all_trip_ids[n_train:n_train + n_val]
test_trips   = all_trip_ids[n_train + n_val:]

df_train = df[df['trip_id'].isin(train_trips)].copy()
df_val   = df[df['trip_id'].isin(val_trips)].copy()
df_test  = df[df['trip_id'].isin(test_trips)].copy()

log(f"Total trips  : {n}")
log(f"Train trips  : {len(train_trips)}  →  {len(df_train):,} rows")
log(f"Val   trips  : {len(val_trips)}  →  {len(df_val):,} rows")
log(f"Test  trips  : {len(test_trips)}  →  {len(df_test):,} rows")
log(f"Train trip IDs: {train_trips[:5]}...{train_trips[-3:]}")
log(f"Val   trip IDs: {val_trips}")
log(f"Test  trip IDs: {test_trips}")


# ══════════════════════════════════════════════
#  STEP 7 — DEFINE FEATURE SETS FOR EACH MODEL
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 7 — Feature sets for each model")
print(SEP)

# ────────────────────────────────────────────
#  VEHICLE DYNAMICS MODEL (XGBoost)
#  Input  → Throttle + previous state
#  Output → Velocity, Acceleration, Motor Torque
# ────────────────────────────────────────────
DYNAMICS_FEATURES = [
    'Throttle [%]',                         # rider intent (real signal)
    'Velocity_kmh_prev',                    # previous speed
    'velocity_ms_prev',                     # previous speed in m/s
    'Longitudinal_Acceleration_ms2_prev',   # previous acceleration
    'Motor_Torque_Nm_prev',                 # previous torque
    'slope',                                # road gradient (derived)
    'Regenerative Braking Signal',          # regen state
    'Ambient Temperature [°C]',             # affects rolling resistance
    'trip_type',                            # A=summer, B=winter (encoded below)
]

DYNAMICS_TARGETS = [
    'Velocity [km/h]',
    'Longitudinal Acceleration [m/s^2]',
    'Motor Torque [Nm]',
]

# ────────────────────────────────────────────
#  RANGE MANAGEMENT MODEL (XGBoost)
#  Input  → Dynamics outputs + battery state
#  Output → SoC(t), energy_norm(t), delta_soc
# ────────────────────────────────────────────
RANGE_FEATURES = [
    'Throttle [%]',                         # rider intent
    'Velocity [km/h]',                      # current speed
    'Longitudinal Acceleration [m/s^2]',    # current acceleration
    'Motor Torque [Nm]',                    # current torque (from dynamics)
    'slope',                                # road gradient
    'Regenerative Braking Signal',          # regen braking active
    'Battery Temperature [°C]',             # battery health factor
    'Ambient Temperature [°C]',             # thermal load
    'SoC_pct_prev',                         # previous SoC
    'energy_norm_Whkg_prev',               # previous energy rate
    'Velocity_kmh_prev',                    # previous speed
    'trip_type',                            # summer/winter encoded
]

RANGE_TARGETS = [
    'SoC [%]',               # current SoC
    'delta_soc',             # SoC change this step
    'energy_norm_Whkg',      # mass-normalised energy (universal)
]

log("Vehicle Dynamics Model (XGBoost)")
log(f"  Features ({len(DYNAMICS_FEATURES)}): {DYNAMICS_FEATURES}")
log(f"  Targets  ({len(DYNAMICS_TARGETS)}): {DYNAMICS_TARGETS}")
log("")
log("Range Management Model (XGBoost)")
log(f"  Features ({len(RANGE_FEATURES)}): {RANGE_FEATURES}")
log(f"  Targets  ({len(RANGE_TARGETS)}): {RANGE_TARGETS}")


# ── Encode trip_type as integer (A=0, B=1) ──
for dset in [df_train, df_val, df_test]:
    dset['trip_type'] = (dset['trip_type'] == 'B').astype(int)


# ══════════════════════════════════════════════
#  STEP 8 — EXPORT CSVs FOR EACH MODEL
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 8 — Export processed CSVs")
print(SEP)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_split(df_tr, df_v, df_te, features, targets, model_name):
    """Save train/val/test CSVs for one model."""
    cols = features + targets + ['trip_id']
    # Keep only columns that exist
    cols = [c for c in cols if c in df_tr.columns]

    for split_name, split_df in [('train', df_tr), ('val', df_v), ('test', df_te)]:
        out_path = os.path.join(OUTPUT_DIR, f"{model_name}_{split_name}.csv")
        split_df[cols].to_csv(out_path, index=False)
        log(f"Saved {model_name}_{split_name}.csv  →  {len(split_df):,} rows  ×  {len(cols)} cols")

save_split(df_train, df_val, df_test,
           DYNAMICS_FEATURES, DYNAMICS_TARGETS,
           'vehicle_dynamics')

save_split(df_train, df_val, df_test,
           RANGE_FEATURES, RANGE_TARGETS,
           'range_management')

# Also save full processed dataset for reference
full_path = os.path.join(OUTPUT_DIR, "processed_bmw_i3_full.csv")
df.to_csv(full_path, index=False)
log(f"\nSaved full processed : processed_bmw_i3_full.csv  →  {len(df):,} rows")


# ══════════════════════════════════════════════
#  STEP 9 — SCOOTER CONVERSION CONSTANTS
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  STEP 9 — Scooter conversion constants")
print(SEP)

log(f"BMW i3 mass used for normalisation : {BMW_MASS} kg")
log(f"E-scooter mass (scooter + rider)   : {SCOOTER_MASS} kg")
log(f"Scale factor (scooter/BMW)         : {SCOOTER_MASS/BMW_MASS:.5f}")
log("")
log("During Digital Twin inference:")
log("  model predicts  → energy_norm_Whkg  (Wh per kg per step)")
log("  scooter energy  = energy_norm_Whkg × SCOOTER_MASS  (Wh per step)")
log("  scooter range   = (SoC/100 × 446 Wh) / (avg energy_Wh per km)")
log("  446 Wh = DualEMobility e-scooter battery capacity")


# ══════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════
print(f"\n{SEP}")
print("  PREPROCESSING COMPLETE")
print(SEP)
print(f"""
  Trips loaded   : {len(all_trip_ids)} total
  Rows processed : {len(df):,}
  Sampling rate  : 10 Hz (Δt = 0.1s)

  Output → {OUTPUT_DIR}/
    vehicle_dynamics_train.csv   ({len(df_train):,} rows)
    vehicle_dynamics_val.csv     ({len(df_val):,} rows)
    vehicle_dynamics_test.csv    ({len(df_test):,} rows)
    range_management_train.csv   ({len(df_train):,} rows)
    range_management_val.csv     ({len(df_val):,} rows)
    range_management_test.csv    ({len(df_test):,} rows)
    processed_bmw_i3_full.csv    (full reference)

  Derived columns:
    velocity_ms          — velocity in m/s
    distance_step_m      — metres per timestep
    slope                — rise/run from elevation (clipped ±0.30)
    power_W              — battery power draw (W)
    energy_step_Wh       — energy per timestep (Wh)
    energy_norm_Whkg     — mass-normalised energy (Wh/kg) ← KEY
    energy_cumulative_Wh — cumulative energy per trip
    delta_soc            — SoC change per timestep
    jerk                 — rate of acceleration change
    *_prev columns       — lagged t-1 state (within trips only)

  Scooter conversion:
    energy_norm_Whkg × {SCOOTER_MASS} kg = scooter Wh per step

  Next steps:
    1. Run preprocess_bmw_i3.py → confirm output
    2. Run model_vehicle_dynamics.py  (XGBoost)
    3. Run model_range_management.py  (XGBoost)
""")
