"""
Synthetic Input Generator — Scenario 5: Winter Cold Start
===========================================================
Cold weather riding — low ambient temp, cautious throttle,
battery performance degraded, winter conditions.
trip_type = 1 (Winter / TripB).

Output: synthetic_inputs/winter_cold_start_inputs.csv

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
OUTPUT_FILE  = "synthetic_inputs/winter_cold_start_inputs.csv"
SEED         = 99

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
print("Generating Winter Cold Start scenario...")
print(f"  Total steps : {TOTAL_STEPS} ({TOTAL_STEPS * DT:.0f} seconds = "
      f"{TOTAL_STEPS * DT / 60:.1f} minutes)")

t = np.arange(TOTAL_STEPS)

# Proportional phase boundaries
cold_start_end = max(2, int(TOTAL_STEPS * 0.06))   # first 6%:  very gentle cold start
warmup_end     = max(cold_start_end + 1,
                     int(TOTAL_STEPS * 0.16))       # next 10%:  warm up
cruise_end     = max(warmup_end + 1,
                     int(TOTAL_STEPS * 0.80))       # next 64%:  normal winter riding
# rest:                                             # last 20%:  return trip

# ── 1. Base throttle — all proportional ──
base_throttle = np.zeros(TOTAL_STEPS)

# Phase 1: Cold start — very gentle
base_throttle[:cold_start_end] = np.linspace(0, 30, cold_start_end)

# Phase 2: Warming up
if warmup_end > cold_start_end:
    base_throttle[cold_start_end:warmup_end] = np.linspace(30, 50,
                                                warmup_end - cold_start_end)

# Phase 3: Normal winter stop-go
# Cycle = 25 seconds = 250 steps, scaled proportionally
cycle_steps = max(10, int(TOTAL_STEPS * 0.05))
accel_end_frac  = 0.40
cruise_end_frac = 0.60
decel_end_frac  = 0.80

for i in range(warmup_end, cruise_end):
    phase = (i - warmup_end) % cycle_steps
    p     = phase / cycle_steps

    if p < accel_end_frac:
        base_throttle[i] = 30 + (p / accel_end_frac) * 35.0
    elif p < cruise_end_frac:
        base_throttle[i] = 50.0
    elif p < decel_end_frac:
        base_throttle[i] = ((decel_end_frac - p) /
                            (decel_end_frac - cruise_end_frac)) * 30.0
    else:
        base_throttle[i] = 0.0

# Phase 4: Return trip — battery slightly warmer
if TOTAL_STEPS > cruise_end:
    base_throttle[cruise_end:] = 45.0

# ── 2. AR(1) noise — smooth (careful winter riding) ──
Z_throttle     = box_muller_normal(TOTAL_STEPS, seed=SEED)
noise_throttle = ar1_walk(Z_throttle, memory=0.94, amplitude=7.0)
throttle       = np.clip(base_throttle + noise_throttle, 0, 100)

# ── 3. Slope — flat winter urban ──
Z_slope     = box_muller_normal(TOTAL_STEPS, seed=SEED + 1)
noise_slope = ar1_walk(Z_slope, memory=0.999, amplitude=0.0006)
slope       = np.clip(noise_slope, -0.04, 0.04)

# ── 4. Regen — conservative (cold battery) ──
regen = np.where(throttle < 5.0, 0.3, 0.0)

# ── 5. Ambient temp — cold, slowly warms ──
Z_temp       = box_muller_normal(TOTAL_STEPS, seed=SEED + 2)
noise_temp   = ar1_walk(Z_temp, memory=0.9999, amplitude=0.03)
temp_trend   = np.linspace(-3.0, 5.0, TOTAL_STEPS)
ambient_temp = np.clip(temp_trend + noise_temp, -10.0, 10.0)

# ── 6. Trip type — WINTER (1) ──
trip_type = np.ones(TOTAL_STEPS, dtype=int)

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
print(f"\n  Trip type                     : 1 (Winter)")
print(f"  Temp range                    : "
      f"{df['ambient_temp'].min():.1f}C to {df['ambient_temp'].max():.1f}C")
print(f"  Below freezing steps          : "
      f"{(df['ambient_temp'] < 0).sum():,}")
print(f"  Cautious throttle (<50%)      : "
      f"{(df['throttle'] < 50).mean()*100:.1f}% of trip")
print(f"\n  NOTE: Range will be lower than summer. Expected winter behaviour.")
print(f"\n  Done.")
