"""
Synthetic Input Generator — Scenario 1: Urban Commute
=======================================================
City riding — stop-go traffic, traffic lights,
moderate throttle, mostly flat road.

Output: synthetic_inputs/urban_commute_inputs.csv

Change TOTAL_STEPS below to any value you want.
All internal ranges scale automatically.
"""

import numpy as np
import pandas as pd
import os

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
TOTAL_STEPS  = 500          # change freely — all ranges scale automatically
DT           = 0.1
OUTPUT_FILE  = "synthetic_inputs/urban_commute_inputs.csv"
SEED         = 42

# ─────────────────────────────────────────────
#  BOX-MULLER GAUSSIAN GENERATOR
# ─────────────────────────────────────────────
def box_muller_normal(n, seed=None):
    rng = np.random.default_rng(seed)
    U1  = rng.uniform(1e-10, 1.0, n)
    U2  = rng.uniform(0.0,   1.0, n)
    return np.sqrt(-2.0 * np.log(U1)) * np.cos(2.0 * np.pi * U2)

def ar1_walk(gaussian_noise, memory=0.95, amplitude=1.0):
    noise = np.zeros(len(gaussian_noise))
    for t in range(1, len(noise)):
        noise[t] = memory * noise[t - 1] + amplitude * gaussian_noise[t]
    return noise

# ─────────────────────────────────────────────
#  SCENARIO
# ─────────────────────────────────────────────
print("Generating Urban Commute scenario...")
print(f"  Total steps : {TOTAL_STEPS} ({TOTAL_STEPS * DT:.0f} seconds = "
      f"{TOTAL_STEPS * DT / 60:.1f} minutes)")

t = np.arange(TOTAL_STEPS)

# ── 1. Base throttle — traffic light cycle ──
# Cycle length proportional to total steps (20 second cycle)
cycle_steps   = max(10, int(TOTAL_STEPS * 0.04))   # 4% of total = ~20s cycle
base_throttle = np.zeros(TOTAL_STEPS)

for i in range(TOTAL_STEPS):
    phase = i % cycle_steps
    accel_end  = int(cycle_steps * 0.40)   # 40% of cycle: accelerate
    cruise_end = int(cycle_steps * 0.60)   # 20% of cycle: cruise
    decel_end  = int(cycle_steps * 0.80)   # 20% of cycle: decelerate
    # rest: stopped

    if phase < accel_end:
        base_throttle[i] = (phase / accel_end) * 55.0
    elif phase < cruise_end:
        base_throttle[i] = 55.0
    elif phase < decel_end:
        base_throttle[i] = ((decel_end - phase) / (decel_end - cruise_end)) * 30.0
    else:
        base_throttle[i] = 0.0

# ── 2. AR(1) noise on throttle ──
Z_throttle     = box_muller_normal(TOTAL_STEPS, seed=SEED)
noise_throttle = ar1_walk(Z_throttle, memory=0.92, amplitude=8.0)
throttle       = np.clip(base_throttle + noise_throttle, 0, 100)

# ── 3. Slope — flat urban ──
Z_slope     = box_muller_normal(TOTAL_STEPS, seed=SEED + 1)
noise_slope = ar1_walk(Z_slope, memory=0.999, amplitude=0.0008)
slope       = np.clip(noise_slope, -0.05, 0.05)

# ── 4. Regen ──
regen = np.where(throttle < 5.0, 0.5, 0.0)

# ── 5. Ambient temp ──
Z_temp       = box_muller_normal(TOTAL_STEPS, seed=SEED + 2)
noise_temp   = ar1_walk(Z_temp, memory=0.9999, amplitude=0.05)
ambient_temp = np.clip(28.0 + noise_temp, 20.0, 38.0)

# ── 6. Trip type ──
trip_type = np.zeros(TOTAL_STEPS, dtype=int)

# ─────────────────────────────────────────────
#  SAVE
# ─────────────────────────────────────────────
df = pd.DataFrame({
    'step'        : t,
    'time_s'      : np.round(t * DT, 2),
    'throttle'    : np.round(throttle, 2),
    'slope'       : np.round(slope, 5),
    'regen'       : np.round(regen, 1),
    'ambient_temp': np.round(ambient_temp, 2),
    'trip_type'   : trip_type,
})

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print(f"\n  Output file : {OUTPUT_FILE}")
print(f"  Rows        : {len(df):,}")
print(f"\n  Input signal statistics:")
print(f"  {'Signal':<20} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8}")
print(f"  {'-'*56}")
for col in ['throttle', 'slope', 'regen', 'ambient_temp']:
    print(f"  {col:<20} {df[col].min():>8.3f} {df[col].max():>8.3f} "
          f"{df[col].mean():>8.3f} {df[col].std():>8.3f}")
print(f"\n  Stopped steps (throttle < 5%)  : "
      f"{(df['throttle'] < 5).sum():,} ({(df['throttle'] < 5).mean()*100:.1f}%)")
print(f"  Regen active steps             : "
      f"{(df['regen'] > 0).sum():,} ({(df['regen'] > 0).mean()*100:.1f}%)")
print(f"\n  Done.")
