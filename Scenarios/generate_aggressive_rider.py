"""
Synthetic Input Generator — Scenario 3: Aggressive Rider
==========================================================
Hard acceleration bursts, rapid throttle changes, high peak throttle.
Designed to push motor toward thermal limits.
Watch rotor temperature during simulation.

Output: synthetic_inputs/aggressive_rider_inputs.csv

Change TOTAL_STEPS below to any value. All ranges scale automatically.
"""

import numpy as np
import pandas as pd
import os

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
TOTAL_STEPS  = 10000
DT           = 0.1
OUTPUT_FILE  = "synthetic_inputs_01/aggressive_rider_inputs.csv"
SEED         = 77

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
print("Generating Aggressive Rider scenario...")
print(f"  Total steps : {TOTAL_STEPS} ({TOTAL_STEPS * DT:.0f} seconds = "
      f"{TOTAL_STEPS * DT / 60:.1f} minutes)")

t = np.arange(TOTAL_STEPS)

# ── 1. Base throttle — burst cycle, all proportional ──
# Cycle = 15 seconds = 150 steps at 10Hz
# Scale cycle proportionally but keep it meaningful
cycle_steps = max(10, int(TOTAL_STEPS * 0.03))  # 3% of total

# Proportional phase fractions within each cycle
launch_frac = 0.07    # 7%  : instant launch
hold_frac   = 0.53    # 53% : hold near-max
release_frac= 0.67    # 67% : sudden release
brake_frac  = 0.87    # 87% : hard braking

base_throttle = np.zeros(TOTAL_STEPS)
for i in range(TOTAL_STEPS):
    phase = i % cycle_steps
    p     = phase / cycle_steps  # normalised 0-1 within cycle

    if p < launch_frac:
        base_throttle[i] = (p / launch_frac) * 95.0
    elif p < hold_frac:
        base_throttle[i] = 90.0 + np.sin(phase * 0.3) * 5.0
    elif p < release_frac:
        base_throttle[i] = ((release_frac - p) / (release_frac - hold_frac)) * 90.0
    elif p < brake_frac:
        base_throttle[i] = 0.0
    else:
        base_throttle[i] = 5.0

base_throttle = np.clip(base_throttle, 0, 100)

# ── 2. AR(1) noise — high amplitude, low memory (jerky) ──
Z_throttle     = box_muller_normal(TOTAL_STEPS, seed=SEED)
noise_throttle = ar1_walk(Z_throttle, memory=0.85, amplitude=18.0)
throttle       = np.clip(base_throttle + noise_throttle, 0, 100)

# ── 3. Slope — mixed terrain ──
Z_slope     = box_muller_normal(TOTAL_STEPS, seed=SEED + 1)
noise_slope = ar1_walk(Z_slope, memory=0.997, amplitude=0.0015)
slope       = np.clip(noise_slope, -0.10, 0.10)

# ── 4. Regen — hard braking threshold ──
regen = np.where(throttle < 10.0, 0.5, 0.0)

# ── 5. Ambient temp — hot ──
Z_temp       = box_muller_normal(TOTAL_STEPS, seed=SEED + 2)
noise_temp   = ar1_walk(Z_temp, memory=0.9999, amplitude=0.06)
ambient_temp = np.clip(35.0 + noise_temp, 28.0, 45.0)

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
print(f"\n  Full throttle steps (>85%)    : "
      f"{(df['throttle'] > 85).sum():,} ({(df['throttle'] > 85).mean()*100:.1f}%)")
print(f"  Hard regen steps              : "
      f"{(df['regen'] > 0).sum():,} ({(df['regen'] > 0).mean()*100:.1f}%)")
print(f"\n  NOTE: Watch rotor temperature — designed to push thermal limits.")
print(f"\n  Done.")
