"""
Synthetic Input Generator — Scenario 2: Highway Cruise
=======================================================
Open road — sustained throttle, gentle hill in the middle,
warm conditions. Tests motor thermal buildup under sustained load.

Output: synthetic_inputs/highway_cruise_inputs.csv

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
OUTPUT_FILE  = "synthetic_inputs/highway_cruise_inputs.csv"
SEED         = 123

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
print("Generating Highway Cruise scenario...")
print(f"  Total steps : {TOTAL_STEPS} ({TOTAL_STEPS * DT:.0f} seconds = "
      f"{TOTAL_STEPS * DT / 60:.1f} minutes)")

t = np.arange(TOTAL_STEPS)

# Proportional phase boundaries
ramp_end    = max(2, int(TOTAL_STEPS * 0.04))    # first 4%: ramp up
decel_start = max(ramp_end + 1, int(TOTAL_STEPS * 0.90))  # last 10%: decelerate

# ── 1. Base throttle ──
base_throttle = np.zeros(TOTAL_STEPS)

# Ramp up
base_throttle[:ramp_end] = np.linspace(0, 72, ramp_end)

# Cruise with gentle variation
cruise_range = np.arange(ramp_end, decel_start)
if len(cruise_range) > 0:
    base_throttle[ramp_end:decel_start] = (
        72.0 + np.sin(cruise_range / max(1, TOTAL_STEPS * 0.06)) * 5.0
    )

# Decelerate
decel_steps = TOTAL_STEPS - decel_start
if decel_steps > 0:
    base_throttle[decel_start:] = np.linspace(72, 0, decel_steps)

base_throttle = np.clip(base_throttle, 0, 100)

# ── 2. AR(1) noise on throttle ──
Z_throttle     = box_muller_normal(TOTAL_STEPS, seed=SEED)
noise_throttle = ar1_walk(Z_throttle, memory=0.97, amplitude=4.0)
throttle       = np.clip(base_throttle + noise_throttle, 0, 100)

# ── 3. Slope — gentle hill in middle ──
Z_slope     = box_muller_normal(TOTAL_STEPS, seed=SEED + 1)
noise_slope = ar1_walk(Z_slope, memory=0.9995, amplitude=0.001)

# Hill occupies steps 30%–60% of the trip
hill_start = int(TOTAL_STEPS * 0.30)
hill_end   = int(TOTAL_STEPS * 0.60)
hill_base  = np.zeros(TOTAL_STEPS)
if hill_end > hill_start:
    hill_base[hill_start:hill_end] = np.sin(
        np.linspace(0, np.pi, hill_end - hill_start)
    ) * 0.04

slope = np.clip(hill_base + noise_slope * 0.5, -0.08, 0.08)

# ── 4. Regen — rare on highway ──
regen = np.where(throttle < 5.0, 0.5, 0.0)

# ── 5. Ambient temp ──
Z_temp       = box_muller_normal(TOTAL_STEPS, seed=SEED + 2)
noise_temp   = ar1_walk(Z_temp, memory=0.9999, amplitude=0.04)
ambient_temp = np.clip(32.0 + noise_temp, 25.0, 40.0)

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
print(f"\n  Cruise steps (throttle > 60%) : "
      f"{(df['throttle'] > 60).sum():,} ({(df['throttle'] > 60).mean()*100:.1f}%)")
print(f"  Hill steps (slope > 0.02)     : "
      f"{(df['slope'] > 0.02).sum():,}")
print(f"\n  Done.")
