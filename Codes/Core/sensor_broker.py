"""
sensor_broker.py
=================
Central broker for E-Scooter Digital Shadow.

KEY FIX: Uses wall-clock time to publish exactly every 1 second,
instead of counting steps. This ensures Unity and Dashboard always
update at a true 1-second interval regardless of simulation speed.

Responsibilities:
  - Runs ALL 4 ML models (app.py no longer does this)
  - Sends state to Unity via TCP port 5005 every 1 real second
  - Writes shared_state.json every 1 real second for app.py to read
  - Both outputs fire from the SAME snapshot at the SAME moment → perfect sync

Run BEFORE app.py and Unity:
    python sensor_broker.py your_scenario.csv [soc_init]

Arguments:
    your_scenario.csv   — scenario CSV file (required)
    soc_init            — starting battery SoC in % (optional, default 80)

Example:
    python sensor_broker.py urban_commute_inputs.csv 90
"""

import sys
import os
import json
import socket
import time
import threading
import numpy as np
import pandas as pd

# ── Add dashboard folder to path so models.py can be imported ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dashboard"))
from models import (
    load_all_models, init_state, run_dt_step,
    SCOOTER_BATTERY_WH, SCOOTER_MASS,
    TEMP_WARN, TEMP_CRITICAL, SOC_WARN, SOC_CRITICAL,
)
from state import init_history, append_history

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
UNITY_HOST    = "127.0.0.1"
UNITY_PORT    = 5005
STATE_FILE    = os.path.join(os.path.dirname(__file__), "dashboard", "Dashboard_v2", "shared_state.json")
STEP_DELAY    = 0.1      # seconds per simulation step (10 Hz model stepping)
PUBLISH_EVERY = 1.0      # publish to Unity + dashboard every 1.0 real seconds
                         # This is time-based, NOT step-count-based.

# ─────────────────────────────────────────────
#  SHARED STATE — written by main thread,
#  read by publisher thread
# ─────────────────────────────────────────────
_latest_snapshot  = {}
_snapshot_lock    = threading.Lock()
_publish_event    = threading.Event()   # signals publisher to fire

# ─────────────────────────────────────────────
#  UNITY SENDER
# ─────────────────────────────────────────────
def _send_to_unity(data: dict):
    """Sends JSON payload to Unity TCPReceiver. Non-blocking on failure."""
    try:
        payload = json.dumps(data).encode("utf-8")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            s.connect((UNITY_HOST, UNITY_PORT))
            s.sendall(payload)
    except Exception:
        pass  # Unity not running — skip silently


# ─────────────────────────────────────────────
#  STATE FILE WRITER
# ─────────────────────────────────────────────
def _write_state_file(data: dict):
    """Writes snapshot to shared_state.json for app.py to read."""
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"[BROKER] State file write error: {e}")


# ─────────────────────────────────────────────
#  SYNCHRONIZED PUBLISHER THREAD
#
#  Waits for publish_event from the simulation loop.
#  When fired: sends IDENTICAL snapshot to Unity AND
#  writes state file at the exact same moment.
#  Both consumers always receive the same data.
# ─────────────────────────────────────────────
def _publisher_thread():
    while True:
        _publish_event.wait()
        _publish_event.clear()

        with _snapshot_lock:
            snapshot = _latest_snapshot.copy()

        if not snapshot:
            continue

        # ── Fire both outputs simultaneously ──
        _send_to_unity(snapshot["unity_payload"])   # → Unity
        _write_state_file(snapshot["dashboard"])    # → app.py

        speed = snapshot["unity_payload"].get("speed", 0)
        soc   = snapshot["unity_payload"].get("soc", 0)
        temp  = snapshot["unity_payload"].get("motor_temp", 0)
        step  = snapshot["dashboard"].get("current_step", 0)
        ts    = time.strftime("%H:%M:%S")
        print(f"[SYNC {ts}] Step={step:5d} | Speed={speed:5.1f} km/h | "
              f"SoC={soc:5.1f}% | Temp={temp:5.1f}°C")


# ─────────────────────────────────────────────
#  BUILD SNAPSHOT FROM CURRENT STATE
# ─────────────────────────────────────────────
def _build_snapshot(state: dict, live_history: dict,
                    full_history: dict, step_idx: int,
                    total_steps: int, scenario_name: str,
                    soc_init: float, data_source: str,
                    sim_complete: bool = False,
                    summary: dict = None) -> dict:
    """
    Builds the two-part snapshot:
      unity_payload  → sent to Unity TCPReceiver
      dashboard      → written to shared_state.json for app.py
    """
    unity_payload = {
        "speed":           float(state.get("velocity", 0)),
        "rpm":             float(state.get("rpm", 0)),
        "motor_temp":      float(state.get("pm_temp", 25)),
        "soc":             float(state.get("soc", 100)),
        "remaining_range": float(state.get("remaining_km", 0)),
        "torque":          float(abs(state.get("torque", 0))),
        "throttle":        float(state.get("throttle", 0)),
        "gps_lat":         float(state.get("gps_lat", 13.6288)),
        "gps_lon":         float(state.get("gps_lon", 79.9611)),
    }

    dashboard = {
        "speed":           unity_payload["speed"],
        "rpm":             unity_payload["rpm"],
        "motor_temp":      unity_payload["motor_temp"],
        "soc":             unity_payload["soc"],
        "remaining_range": unity_payload["remaining_range"],
        "torque":          unity_payload["torque"],
        "throttle":        unity_payload["throttle"],
        "accel":           float(state.get("accel", 0)),
        "stator_temp":     float(state.get("stator_temp", 25)),
        "v_phase":         float(state.get("v_phase", 0)),
        "u_d":             float(state.get("u_d", 0)),
        "u_q":             float(state.get("u_q", 0)),
        "i_d":             float(state.get("i_d", 0)),
        "i_q":             float(state.get("i_q", 0)),
        "duty_cycle":      float(state.get("duty_cycle", 0)),
        "energy_norm":     float(state.get("energy_norm", 0)),
        "gps_lat":         unity_payload["gps_lat"],
        "gps_lon":         unity_payload["gps_lon"],

        "scenario_name":   scenario_name,
        "total_steps":     total_steps,
        "current_step":    step_idx + 1,
        "soc_init":        soc_init,
        "elapsed_s":       round((step_idx + 1) * 0.1, 1),
        "data_source":     data_source,
        "sim_complete":    sim_complete,
        "timestamp":       time.time(),

        "history":         live_history,
    }

    if sim_complete and summary:
        dashboard["summary"]      = summary
        dashboard["full_history"] = full_history

    return {"unity_payload": unity_payload, "dashboard": dashboard}


# ─────────────────────────────────────────────
#  COMPUTE FINAL SUMMARY
# ─────────────────────────────────────────────
def _compute_summary(full_history: dict, soc_init: float,
                     total_steps: int, scenario_name: str) -> dict:
    h       = full_history
    soc_arr = np.array(h["soc"])
    vel_arr = np.array(h["velocity"])
    pm_arr  = np.array(h["pm_temp"])
    rg_arr  = np.array(h["regen"])

    total_dist = float(np.sum(vel_arr / 3.6 * 0.1 / 1000))
    soc_used   = float(soc_init - soc_arr[-1])

    soc_shifted = np.concatenate([[float(soc_init)], soc_arr[:-1]])
    soc_drops   = np.maximum(soc_shifted - soc_arr, 0.0)
    soc_gains   = np.maximum(soc_arr - soc_shifted, 0.0)
    energy_wh_s = soc_drops / 100.0 * SCOOTER_BATTERY_WH
    regen_wh_a  = soc_gains / 100.0 * SCOOTER_BATTERY_WH

    energy_wh   = float(np.sum(energy_wh_s))
    regen_wh    = float(np.sum(regen_wh_a[rg_arr > 0]))
    peak_temp   = float(np.max(pm_arr))
    t_warn      = int(np.sum(pm_arr >= TEMP_WARN)) * 0.1
    t_crit      = int(np.sum(pm_arr >= TEMP_CRITICAL)) * 0.1
    avg_spd     = float(np.mean(vel_arr[vel_arr > 0.5])) if np.any(vel_arr > 0.5) else 0.0
    final_range = float(h["remaining_km"][-1])
    avg_eff     = energy_wh / max(total_dist, 0.001)

    return {
        "scenario_name":        scenario_name,
        "total_steps":          total_steps,
        "duration_s":           total_steps * 0.1,
        "total_distance_km":    total_dist,
        "soc_consumed_pct":     soc_used,
        "soc_final":            float(soc_arr[-1]),
        "soc_init":             soc_init,
        "energy_consumed_wh":   energy_wh,
        "regen_recovered_wh":   regen_wh,
        "net_energy_wh":        energy_wh - regen_wh,
        "avg_efficiency_wh_km": avg_eff,
        "peak_rotor_temp":      peak_temp,
        "time_above_warn_s":    t_warn,
        "time_critical_s":      t_crit,
        "avg_speed_kmh":        avg_spd,
        "final_range_km":       final_range,
    }


# ─────────────────────────────────────────────
#  CSV SIMULATION MODE
#
#  KEY CHANGE FROM OLD VERSION:
#  Instead of publishing every SYNC_EVERY (10) steps,
#  we now publish based on WALL-CLOCK TIME using
#  time.monotonic(). This guarantees exactly 1 real
#  second between each Unity + Dashboard update,
#  regardless of how many steps have run.
# ─────────────────────────────────────────────
def run_csv_mode(models: dict, csv_path: str, soc_init: float = 80.0):
    if not os.path.exists(csv_path):
        print(f"[BROKER] ERROR: CSV not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    required = ["throttle", "slope", "regen", "ambient_temp", "trip_type"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"[BROKER] ERROR: CSV missing columns: {missing}")
        sys.exit(1)

    scenario_name = (os.path.basename(csv_path)
                     .replace("_inputs.csv", "")
                     .replace("_", " ").title())
    total_steps = len(df)

    print(f"[BROKER] Scenario   : {scenario_name}")
    print(f"[BROKER] Steps      : {total_steps} ({total_steps * 0.1:.0f}s simulated)")
    print(f"[BROKER] Starting SoC: {soc_init}%")
    print(f"[BROKER] Step delay : {STEP_DELAY}s ({1/STEP_DELAY:.0f} Hz)")
    print(f"[BROKER] Sync rate  : every {PUBLISH_EVERY}s — Unity + Dashboard update together")
    print("[BROKER] Starting simulation...\n")

    first_ambient = float(df["ambient_temp"].iloc[0])
    state         = init_state(models, first_ambient, soc_init)
    live_history  = init_history()

    full_history = {k: [] for k in [
        "step", "throttle", "slope", "regen",
        "u_d", "u_q", "v_phase", "duty_cycle",
        "i_d", "i_q", "torque", "rpm",
        "pm_temp", "stator_temp",
        "velocity", "accel",
        "soc", "remaining_km",
        "energy_norm", "battery_temp_display",
    ]}

    # ── Time-based publish tracking ──
    # We record when the last publish happened using wall-clock time.
    # Every time PUBLISH_EVERY seconds have elapsed, we trigger a sync.
    last_publish_time = time.monotonic()

    for step_idx, row in df.iterrows():
        step_start = time.monotonic()

        throttle  = float(row["throttle"])
        slope     = float(row["slope"])
        regen     = float(row["regen"])
        ambient   = float(row["ambient_temp"])
        trip_type = int(row["trip_type"])

        # ── Run all 4 ML models ──
        state = run_dt_step(models, state, throttle, slope,
                            ambient, trip_type, regen)

        # Pass through GPS if CSV has it
        state["gps_lat"] = float(row.get("gps_lat", 13.6288))
        state["gps_lon"] = float(row.get("gps_lon", 79.9611))

        # ── Append to histories ──
        live_history = append_history(live_history, step_idx, state)

        full_history["step"].append(step_idx)
        full_history["throttle"].append(throttle)
        full_history["slope"].append(slope)
        full_history["regen"].append(regen)
        for k in ["u_d", "u_q", "v_phase", "duty_cycle",
                  "i_d", "i_q", "torque", "rpm",
                  "pm_temp", "stator_temp",
                  "velocity", "accel",
                  "soc", "remaining_km",
                  "energy_norm", "battery_temp_display"]:
            full_history[k].append(state.get(k, 0.0))

        # ── Time-based publish: fire every PUBLISH_EVERY seconds ──
        now = time.monotonic()
        time_since_last = now - last_publish_time

        is_last_step = (step_idx == total_steps - 1)

        if time_since_last >= PUBLISH_EVERY or is_last_step:
            snapshot = _build_snapshot(
                state=state,
                live_history=live_history,
                full_history=full_history,
                step_idx=step_idx,
                total_steps=total_steps,
                scenario_name=scenario_name,
                soc_init=soc_init,
                data_source="csv",
                sim_complete=False,
            )
            with _snapshot_lock:
                _latest_snapshot.update(snapshot)
            _publish_event.set()           # wake publisher thread
            last_publish_time = now        # reset timer

        # ── Precise step timing ──
        # Sleep only the remaining time in this step's budget.
        # This prevents drift accumulation over many steps.
        elapsed_in_step = time.monotonic() - step_start
        sleep_time = STEP_DELAY - elapsed_in_step
        if sleep_time > 0:
            time.sleep(sleep_time)

    # ── Final publish with complete summary ──
    print("\n[BROKER] Simulation complete. Computing summary...")
    summary = _compute_summary(full_history, soc_init,
                               total_steps, scenario_name)

    final_snapshot = _build_snapshot(
        state=state,
        live_history=live_history,
        full_history=full_history,
        step_idx=total_steps - 1,
        total_steps=total_steps,
        scenario_name=scenario_name,
        soc_init=soc_init,
        data_source="csv",
        sim_complete=True,
        summary=summary,
    )
    with _snapshot_lock:
        _latest_snapshot.update(final_snapshot)
    _publish_event.set()

    time.sleep(0.5)  # give publisher time to write final state
    print("[BROKER] Done. Dashboard and Unity updated with final state.")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sensor_broker.py your_scenario.csv [soc_init]")
        print("Example: python sensor_broker.py urban_commute_inputs.csv 90")
        sys.exit(1)

    csv_path = sys.argv[1]
    soc_init = float(sys.argv[2]) if len(sys.argv) > 2 else 80.0

    # ── Load ML models ──
    print("[BROKER] Loading ML models...")
    models, err = load_all_models()
    if err:
        print(f"[BROKER] ERROR loading models: {err}")
        sys.exit(1)
    print("[BROKER] All 4 models loaded successfully.")

    # ── Start synchronized publisher thread ──
    pub_thread = threading.Thread(target=_publisher_thread, daemon=True)
    pub_thread.start()
    print(f"[BROKER] Publisher started → Unity port {UNITY_PORT} + state file")
    print(f"[BROKER] Unity and Dashboard will update in sync every {PUBLISH_EVERY}s")
    print()

    # ── Run simulation ──
    run_csv_mode(models, csv_path, soc_init)
