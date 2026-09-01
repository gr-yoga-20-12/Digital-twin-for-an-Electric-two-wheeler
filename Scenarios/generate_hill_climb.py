"""
Synthetic Input Generator — Scenario 4: Hill Climb & Descent
=============================================================
Repeated hill climb and descent cycles.
Tests motor torque under gradient load and regen on descent.

Output: synthetic_inputs/hill_climb_inputs.csv

Change TOTAL_STEPS below to any value. All ranges scale automatically.
"""

import numpy as np
import pandas as pd
import os

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
TOTAL_STEPS  = 500
DT           = 0.1
OUTPUT_FILE  = "synthetic_inputs/hill_climb_inputs.csv"
SEED         = 55

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
print("Generating Hill Climb & Descent scenario...")
print(f"  Total steps : {TOTAL_STEPS} ({TOTAL_STEPS * DT:.0f} seconds = "
      f"{TOTAL_STEPS * DT / 60:.1f} minutes)")

t = np.arange(TOTAL_STEPS)

# ── 1. Slope profile — proportional hill cycles ──
# 5 hill cycles across the full trip
n_cycles   = 5
hill_cycle = max(10, TOTAL_STEPS // n_cycles)

# Proportional phase fractions within each hill cycle
flat_frac    = 0.10   # 10% flat approach
rising_frac  = 0.30   # 20% rising slope
steep_frac   = 0.50   # 20% sustained steep
peak_frac    = 0.65   # 15% approaching peak / going over
descent_frac = 0.85   # 20% descent
# rest: flattening back

slope_base = np.zeros(TOTAL_STEPS)
for i in range(TOTAL_STEPS):
    phase = i % hill_cycle
    p     = phase / hill_cycle   # normalised 0-1

    if p < flat_frac:
        slope_base[i] = 0.01
    elif p < rising_frac:
        frac = (p - flat_frac) / (rising_frac - flat_frac)
        slope_base[i] = 0.01 + frac * 0.14
    elif p < steep_frac:
        slope_base[i] = 0.15
    elif p < peak_frac:
        frac = (p - steep_frac) / (peak_frac - steep_frac)
        slope_base[i] = 0.15 - frac * 0.23
    elif p < descent_frac:
        slope_base[i] = -0.08
    else:
        frac = (p - descent_frac) / (1.0 - descent_frac)
        slope_base[i] = -0.08 + frac * 0.09

# AR(1) noise on slope
Z_slope     = box_muller_normal(TOTAL_STEPS, seed=SEED + 1)
noise_slope = ar1_walk(Z_slope, memory=0.998, amplitude=0.0008)
slope       = np.clip(slope_base + noise_slope, -0.20, 0.20)

# ── 2. Base throttle — follows slope ──
base_throttle = np.zeros(TOTAL_STEPS)
for i in range(TOTAL_STEPS):
    s = slope[i]
    if   s > 0.08:  base_throttle[i] = 85.0
    elif s > 0.03:  base_throttle[i] = 65.0
    elif s > -0.02: base_throttle[i] = 45.0
    elif s > -0.06: base_throttle[i] = 15.0
    else:           base_throttle[i] = 0.0

# ── 3. AR(1) noise on throttle ──
Z_throttle     = box_muller_normal(TOTAL_STEPS, seed=SEED)
noise_throttle = ar1_walk(Z_throttle, memory=0.90, amplitude=10.0)
throttle       = np.clip(base_throttle + noise_throttle, 0, 100)

# ── 4. Regen — active on downhill ──
regen = np.zeros(TOTAL_STEPS)
for i in range(TOTAL_STEPS):
    if throttle[i] < 5.0 and slope[i] < -0.03:
        regen[i] = 0.5
    elif throttle[i] < 5.0:
        regen[i] = 0.3
    else:
        regen[i] = 0.0

# ── 5. Ambient temp — warms through trip ──
Z_temp       = box_muller_normal(TOTAL_STEPS, seed=SEED + 2)
noise_temp   = ar1_walk(Z_temp, memory=0.9999, amplitude=0.03)
temp_trend   = np.linspace(22.0, 30.0, TOTAL_STEPS)
ambient_temp = np.clip(temp_trend + noise_temp, 15.0, 38.0)

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
    'regen'       : np.round(regen, 2),
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
hill_cycles_actual = TOTAL_STEPS // hill_cycle
print(f"\n  Hill cycles in trip           : {hill_cycles_actual}")
print(f"  Steep climb steps (slope>0.10): "
      f"{(df['slope'] > 0.10).sum():,} ({(df['slope'] > 0.10).mean()*100:.1f}%)")
print(f"  Descent steps (slope<-0.03)   : "
      f"{(df['slope'] < -0.03).sum():,} ({(df['slope'] < -0.03).mean()*100:.1f}%)")
print(f"  Regen active steps            : "
      f"{(df['regen'] > 0).sum():,} ({(df['regen'] > 0).mean()*100:.1f}%)")
print(f"\n  Done.")
