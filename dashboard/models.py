"""
models.py — Model loading and Digital Twin step execution
==========================================================
All 4 subsystem models loaded and run in feedback loop.

Model architecture:
  Motor Controller : XGBoost (original)
  Motor Performance: LSTM seq=50 (original)
  Vehicle Dynamics : LSTM seq=150 — NEW (replaces XGBoost)
  Range Management : LSTM seq=200 — NEW (replaces XGBoost)

Key design decisions:
  - Dynamics LSTM uses 150-step (15s) driving history
  - Range LSTM uses 200-step (20s) battery drain history
  - Dynamics model predicts in BMW i3 scale, then scaled to scooter
  - Range model predicts SoC AND remaining_km simultaneously
  - SoC monotonicity enforced post-prediction
"""

import numpy as np
import joblib
import os
import streamlit as st

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
MODEL_DIR          = "Models"
SCOOTER_MASS       = 90.0
SCOOTER_BATTERY_WH = 446.0
BMW_MASS           = 1270.0
TEMP_WARN          = 80.0
TEMP_CRITICAL      = 100.0
SOC_WARN           = 25.0
SOC_CRITICAL       = 10.0

# Velocity scaling: BMW i3 max -> scooter max
BMW_MAX_SPEED      = 150.0
SCOOTER_MAX_SPEED  =  25.0
VELOCITY_SCALE     = SCOOTER_MAX_SPEED / BMW_MAX_SPEED   # = 0.1667


# ─────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    """
    Loads all 4 trained models and their scalers.
    Cached — only runs once per Streamlit session.

    Dynamics and Range use NEW LSTM models.
    Motor Controller and Motor Performance unchanged.

    Returns (models_dict, error_string_or_None).
    """
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')

        # ── Motor Controller (XGBoost — unchanged) ──
        ctrl_model  = joblib.load(
            os.path.join(MODEL_DIR, "controller_model.joblib"))
        ctrl_scaler = joblib.load(
            os.path.join(MODEL_DIR, "controller_scaler.joblib"))

        # ── Motor Performance (LSTM seq=50 — unchanged) ──
        motor_model   = tf.keras.models.load_model(
            os.path.join(MODEL_DIR, "motor_model.keras"))
        motor_feat_sc = joblib.load(
            os.path.join(MODEL_DIR, "motor_feat_scaler.joblib"))
        motor_tgt_sc  = joblib.load(
            os.path.join(MODEL_DIR, "motor_tgt_scaler.joblib"))
        motor_config  = joblib.load(
            os.path.join(MODEL_DIR, "motor_config.joblib"))

        # ── Vehicle Dynamics (LSTM seq=150 — NEW) ──
        dyn_model     = tf.keras.models.load_model(
            os.path.join(MODEL_DIR, "dynamics_lstm_model.keras"))
        dyn_feat_sc   = joblib.load(
            os.path.join(MODEL_DIR, "dynamics_lstm_feat_scaler.joblib"))
        dyn_tgt_sc    = joblib.load(
            os.path.join(MODEL_DIR, "dynamics_lstm_tgt_scaler.joblib"))
        dyn_config    = joblib.load(
            os.path.join(MODEL_DIR, "dynamics_lstm_config.joblib"))

        # ── Range Management (LSTM seq=200 — NEW) ──
        rng_model     = tf.keras.models.load_model(
            os.path.join(MODEL_DIR, "range_lstm_model.keras"))
        rng_feat_sc   = joblib.load(
            os.path.join(MODEL_DIR, "range_lstm_feat_scaler.joblib"))
        rng_tgt_sc    = joblib.load(
            os.path.join(MODEL_DIR, "range_lstm_tgt_scaler.joblib"))
        rng_config    = joblib.load(
            os.path.join(MODEL_DIR, "range_lstm_config.joblib"))

        return {
            # Motor Controller
            'ctrl_model'   : ctrl_model,
            'ctrl_scaler'  : ctrl_scaler,
            # Motor Performance
            'motor_model'  : motor_model,
            'motor_feat_sc': motor_feat_sc,
            'motor_tgt_sc' : motor_tgt_sc,
            'motor_config' : motor_config,
            # Vehicle Dynamics LSTM
            'dyn_model'    : dyn_model,
            'dyn_feat_sc'  : dyn_feat_sc,
            'dyn_tgt_sc'   : dyn_tgt_sc,
            'dyn_config'   : dyn_config,
            # Range Management LSTM
            'rng_model'    : rng_model,
            'rng_feat_sc'  : rng_feat_sc,
            'rng_tgt_sc'   : rng_tgt_sc,
            'rng_config'   : rng_config,
        }, None

    except Exception as e:
        return None, str(e)


# ─────────────────────────────────────────────
#  INITIAL STATE
# ─────────────────────────────────────────────
def init_state(models, ambient, soc_start):
    """
    Creates the initial Digital Twin state dictionary.

    Key additions vs XGBoost version:
      - dyn_seq_buffer  : rolling 150-step window for Dynamics LSTM
      - rng_seq_buffer  : rolling 200-step window for Range LSTM
      - velocity_raw_prev: BMW-scale velocity (fed into model)
      - velocity_prev   : scooter-scale velocity (for display)
    """
    motor_seq_len = models['motor_config']['seq_len']           # 50
    dyn_seq_len   = models['dyn_config']['seq_len']             # 150
    rng_seq_len   = models['rng_config']['seq_len']             # 200
    dyn_n_feat    = len(models['dyn_config']['seq_features'])   # 6
    rng_n_feat    = len(models['rng_config']['seq_features'])   # 6

    return {
        # ── Controller feedback ──
        'throttle_proxy'     : 0.0,
        'i_d_prev'           : 0.0,
        'i_q_prev'           : 0.0,
        'motor_speed_prev'   : 0.0,
        'pm_prev'            : float(ambient),
        'stator_winding_prev': float(ambient),
        'stator_tooth_prev'  : float(ambient),
        'stator_yoke_prev'   : float(ambient),
        'duty_cycle_prev'    : 0.0,
        't_load'             : 0.0,

        # ── Dynamics feedback ──
        # velocity_raw_prev: BMW-scale (0-150 km/h)
        #   fed back into Dynamics LSTM as input (scaler fitted on BMW data)
        # velocity_prev: scooter-scale (0-25 km/h)
        #   used for display and range formula only
        'velocity_raw_prev'  : 0.0,
        'velocity_prev'      : 0.0,
        'accel_raw_prev'     : 0.0,
        'accel_prev'         : 0.0,
        'torque_prev'        : 0.0,
        'torque_raw_prev'    : 0.0,

        # ── Range feedback ──
        'soc_prev'           : float(soc_start),
        'energy_norm_prev'   : 0.0,
        'battery_temp'       : float(ambient),

        # ── LSTM rolling sequence buffers ──
        # Motor: 50 steps x 12 features
        'motor_seq_buffer'   : np.zeros(
            (motor_seq_len, 12), dtype=np.float32),
        # Add this flag right after:
        '_motor_prefill' : True,
        # Dynamics: 150 steps x 6 features (all zeros = scooter at rest)
        'dyn_seq_buffer'     : np.zeros(
            (dyn_seq_len, dyn_n_feat), dtype=np.float32),
        # Range: 200 steps x 6 features
        # Pre-fill with initial SoC so LSTM has meaningful context from step 0
        # Avoids SoC starting at wrong value due to zero-buffer warmup
        'rng_seq_buffer'     : np.zeros(
            (rng_seq_len, rng_n_feat), dtype=np.float32),
        # Note: buffer is pre-filled in first call to run_dt_step via _rng_prefill flag
        '_rng_prefill'       : True,

        # ── Cumulatives for range cross-check ──
        'cum_wh'             : 0.0,
        'cum_km'             : 0.0,

        # ── Display outputs (updated each step) ──
        'u_d'  : 0.0, 'u_q'  : 0.0,
        'i_d'  : 0.0, 'i_q'  : 0.0,
        'torque': 0.0, 'rpm'  : 0.0,
        'pm_temp'     : float(ambient),
        'stator_temp' : float(ambient),
        'velocity'    : 0.0,
        'accel'       : 0.0,
        'soc'         : float(soc_start),
        'energy_norm' : 0.0,
        'remaining_km': float(
            soc_start / 100.0 * SCOOTER_BATTERY_WH / 16.6),
        'duty_cycle'  : 0.0,
        'v_phase'     : 0.0,
        'slope'       : 0.0,
        'regen'       : 0.0,
        'throttle'    : 0.0,
        'battery_temp_display': float(ambient),
    }


# ─────────────────────────────────────────────
#  SINGLE DT STEP
# ─────────────────────────────────────────────
def run_dt_step(models, state, throttle, slope,
                ambient, trip_type, regen):
    """
    Executes one complete Digital Twin timestep through all 4 models.

    Flow:
      throttle + motor_state
          -> Model 1: Controller (XGBoost)      -> u_d, u_q
          -> Model 2: Motor LSTM (seq=50)        -> i, torque, rpm, temps
          -> Model 3: Dynamics LSTM (seq=150)    -> velocity, accel, torque
          -> Model 4: Range LSTM (seq=200)       -> SoC, remaining_km

    Key design:
      Dynamics LSTM uses BMW-scale velocity in its sequence buffer.
      The output velocity_raw is in BMW scale (0-150 km/h).
      It is scaled to scooter range (0-25 km/h) for display.

      Range LSTM sequence contains SoC history — the model sees
      exactly how SoC has been falling over the last 20 seconds.
      From this pattern it predicts the next SoC AND remaining_km.

      SoC monotonicity: enforced post-prediction.
        If no regen: SoC(t) = min(predicted, SoC(t-1))
        If regen:    SoC(t) = min(predicted, SoC(t-1) + 0.5)
    """
    s  = state.copy()
    DT = 0.1  # seconds per step

    # ══════════════════════════════════════════
    #  MODEL 1: MOTOR CONTROLLER (XGBoost)
    # ══════════════════════════════════════════
    ctrl_input = np.array([[
        s['throttle_proxy'],
        s['i_d_prev'], s['i_q_prev'],
        s['motor_speed_prev'],
        s['pm_prev'], s['stator_winding_prev'],
        ambient,
        s['duty_cycle_prev'],
    ]])
    ctrl_sc   = models['ctrl_scaler'].transform(ctrl_input)
    ctrl_pred = models['ctrl_model'].predict(ctrl_sc)
    u_d = float(ctrl_pred[0, 0])
    u_q = float(ctrl_pred[0, 1])

    # ══════════════════════════════════════════
    #  MODEL 2: MOTOR PERFORMANCE (LSTM seq=50)
    # ══════════════════════════════════════════
    motor_row = np.array([[
        u_d, u_q,
        s['i_d_prev'], s['i_q_prev'],
        s['motor_speed_prev'],
        s['motor_speed_prev'] * (2 * np.pi / 60),
        s['pm_prev'], s['stator_winding_prev'],
        s['stator_tooth_prev'], s['stator_yoke_prev'],
        ambient, s['t_load'],
    ]])
    motor_row_sc = models['motor_feat_sc'].transform(motor_row)

    # ── Pre-fill Motor buffer on very first step ──
    if s.get('_motor_prefill', False):
        idle_motor_row = np.array([[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            float(s['pm_prev']),
            float(s['stator_winding_prev']),
            float(s['stator_tooth_prev']),
            float(s['stator_yoke_prev']),
            float(ambient),
            0.0,
        ]])
        idle_sc = models['motor_feat_sc'].transform(idle_motor_row)
        s['motor_seq_buffer'][:] = idle_sc[0]
        s['_motor_prefill'] = False

    # NOW copy after prefill is done
    motor_buf = s['motor_seq_buffer'].copy()
    motor_buf = np.roll(motor_buf, -1, axis=0)
    motor_buf[-1] = motor_row_sc[0]
    s['motor_seq_buffer'] = motor_buf

    motor_pred_sc = models['motor_model'].predict(
        motor_buf[np.newaxis, :, :], verbose=0)
    motor_pred = models['motor_tgt_sc'].inverse_transform(
        motor_pred_sc)[0]

    i_d      = float(motor_pred[0])
    i_q      = float(motor_pred[1])
    torque   = float(motor_pred[2])
    rpm      = float(np.clip(motor_pred[3], 0.0, 6000.0))
    pm_temp  = float(motor_pred[4])
    stator_t = float(motor_pred[5])

    # ══════════════════════════════════════════
    #  MODEL 3: VEHICLE DYNAMICS (LSTM seq=150)
    # ══════════════════════════════════════════
    # The Dynamics LSTM sequence buffer stores BMW-scale signals.
    # velocity_raw_prev is BMW-scale — matches training distribution.
    # The scaler was fitted on BMW i3 data so BMW-scale inputs
    # produce correctly normalised z-scores.
    dyn_row = np.array([[
        throttle,                   # [0] Throttle [%]
        s['velocity_raw_prev'],     # [1] Velocity [km/h]  — BMW scale, matches training
        s['accel_raw_prev'],        # [2] Longitudinal Acceleration [m/s^2]
        s['torque_raw_prev'],        # [3] Motor Torque [Nm] — BMW scale matches training
        slope,                      # [4] slope
        regen,                      # [5] Regenerative Braking Signal
    ]])
    dyn_row_sc= models['dyn_feat_sc'].transform(dyn_row)

    # Roll Dynamics LSTM buffer
    dyn_buf     = s['dyn_seq_buffer'].copy()
    dyn_buf     = np.roll(dyn_buf, -1, axis=0)
    dyn_buf[-1] = dyn_row_sc[0]
    s['dyn_seq_buffer'] = dyn_buf

    dyn_pred_sc = models['dyn_model'].predict(
        dyn_buf[np.newaxis, :, :], verbose=0)
    dyn_pred    = models['dyn_tgt_sc'].inverse_transform(
        dyn_pred_sc)[0]

    # Raw outputs in BMW i3 scale
    velocity_raw = float(np.clip(dyn_pred[0], 0.0, BMW_MAX_SPEED))
    accel_raw    = float(dyn_pred[1])
    dyn_torq_raw = float(dyn_pred[2])

    # Scale to e-scooter operating range
    velocity = float(np.clip(velocity_raw * VELOCITY_SCALE,
                              0.0, SCOOTER_MAX_SPEED))
    accel    = float(accel_raw * VELOCITY_SCALE)
    dyn_torq = float(dyn_torq_raw * (SCOOTER_MASS / BMW_MASS))

    # ── Battery temperature thermal model ──
    energy_load  = abs(i_q) * 0.001
    cooling_rate = 0.002
    new_bat_temp = float(np.clip(
        s['battery_temp']
        + energy_load
        - cooling_rate * (s['battery_temp'] - ambient),
        ambient, 60.0))

    # ══════════════════════════════════════════
    #  MODEL 4: RANGE MANAGEMENT (LSTM seq=200)
    # ══════════════════════════════════════════
    # The Range LSTM sequence buffer stores:
    #   SoC history, energy rate, velocity (BMW scale),
    #   throttle, slope, battery temperature
    # SoC and energy history is the most critical input —
    # the model sees HOW SoC has been falling for 20 seconds.
    #
    # velocity_raw_prev used here (BMW scale) because
    # the Range scaler was fitted on BMW i3 velocity values.
    rng_row = np.array([[
        s['soc_prev'],
        s['energy_norm_prev'],
        s['velocity_raw_prev'],   # BMW scale — matches training
        throttle,
        slope,
        new_bat_temp,
    ]])
    rng_row_sc = models['rng_feat_sc'].transform(rng_row)

    # ── Pre-fill Range buffer on first step ──
    # On very first call, fill entire buffer with initial SoC row
    # This prevents the LSTM from predicting with a zero-context buffer
    if s.get('_rng_prefill', False):
        # Build a representative initial row using starting state
        init_rng_row = np.array([[
            s['soc_prev'], 0.0, 0.0, 0.0, 0.0, new_bat_temp
        ]])
        init_rng_sc = models['rng_feat_sc'].transform(init_rng_row)
        s['rng_seq_buffer'][:] = init_rng_sc[0]   # fill all 200 rows
        s['_rng_prefill'] = False

    # Roll Range LSTM buffer
    rng_buf     = s['rng_seq_buffer'].copy()
    rng_buf     = np.roll(rng_buf, -1, axis=0)
    rng_buf[-1] = rng_row_sc[0]
    s['rng_seq_buffer'] = rng_buf

    rng_pred_sc = models['rng_model'].predict(
        rng_buf[np.newaxis, :, :], verbose=0)
    rng_pred    = models['rng_tgt_sc'].inverse_transform(
        rng_pred_sc)[0]

    # Output 0: SoC [%]
    # Output 1: remaining_km
    soc_raw      = float(rng_pred[0])
    remaining_km = float(np.clip(rng_pred[1], 0.0, 100.0))

    # ── SoC monotonicity enforcement ──
    # Battery SoC can only decrease during discharge.
    # It may increase slightly during regen braking.
    # This is a hard physical constraint applied post-prediction.
    if regen > 0:
        # Regen allowed — small SoC recovery possible
        soc = float(np.clip(soc_raw, 0.0,
                             s['soc_prev'] + 0.5))
    else:
        # No regen — SoC must not increase
        soc = float(np.clip(soc_raw, 0.0, s['soc_prev']))

    # Also clip remaining_km to be consistent with SoC
    # (if SoC is low, remaining_km cannot be high)
    max_possible_range = (soc / 100.0 * SCOOTER_BATTERY_WH) / 5.0
    remaining_km = float(np.clip(remaining_km, 0.0, max_possible_range))

    # ── Energy norm for history tracking ──
    # Signed energy: positive = discharging, negative = charging (regen)
    # This MUST preserve sign to match training data convention.
    # In training data (BMW i3): energy_norm_Whkg is negative when
    # battery current is positive (regen feeding energy back in).
    # The Range LSTM learned this negative signal = charging.
    # If we clip to 0, we break that learned association.
    soc_delta    = s['soc_prev'] - soc          # positive=discharge, negative=charging
    energy_wh    = soc_delta / 100.0 * SCOOTER_BATTERY_WH   # Wh, signed
    energy_norm  = energy_wh / SCOOTER_MASS     # Wh/kg, signed — matches training

    # ── Derived controller values ──
    v_phase   = float(np.sqrt(u_d**2 + u_q**2))
    # Paderborn PMSM peak phase voltage ~150V (not 48V scooter battery)
    # duty cycle = fraction of maximum motor voltage being used
    MOTOR_PEAK_V = 150.0
    duty      = float(np.clip(v_phase / MOTOR_PEAK_V, 0.0, 1.0))
    # ── throttle_proxy: use ACTUAL throttle, not i_q feedback ──
    # CRITICAL FIX: The old formula (i_q/250*100) created a locked braking loop.
    # i_q starts negative (cold buffer) -> proxy negative -> Controller predicts
    # braking voltages -> Motor predicts negative i_q -> proxy stays negative.
    # The motor was NEVER able to exit braking mode.
    #
    # The correct approach: use the actual throttle input from the CSV.
    # In the Paderborn dataset, i_q WAS the throttle proxy because there was
    # no separate throttle signal. In our Digital Twin, we have an explicit
    # throttle input from the rider — that is the correct signal to use.
    #
    # When regen > 0: send negative proxy so Controller predicts regen braking
    # When regen == 0: send actual throttle (positive) so Controller drives motor
    if regen > 0:
        thr_proxy = float(-regen * 50.0)          # negative = braking command
    else:
        thr_proxy = float(np.clip(throttle, 0.0, 100.0))  # actual rider intent
    omega     = rpm * (2.0 * np.pi / 60.0)
    t_load    = float(torque
                       - 0.0011 * omega / max(0.1, 0.5)
                       - 0.0015 * omega)

    # ══════════════════════════════════════════
    #  RETURN UPDATED STATE
    # ══════════════════════════════════════════
    return {
        # Controller feedback
        'throttle_proxy'     : thr_proxy,
        'i_d_prev'           : i_d,
        'i_q_prev'           : i_q,
        'motor_speed_prev'   : rpm,
        'pm_prev'            : pm_temp,
        'stator_winding_prev': stator_t,
        'stator_tooth_prev'  : s['stator_tooth_prev'],
        'stator_yoke_prev'   : s['stator_yoke_prev'],
        'duty_cycle_prev'    : duty,
        't_load'             : t_load,

        # Dynamics feedback
        'velocity_raw_prev'  : velocity_raw,  # BMW scale → Dynamics input
        'velocity_prev'      : velocity,       # scooter scale → display
        'accel_raw_prev'     : accel_raw,
        'accel_prev'         : accel,
        'torque_prev'        : dyn_torq,       # scooter scale (display only)
        'torque_raw_prev'    : dyn_torq_raw,   # BMW scale (Dynamics LSTM input)

        # Range feedback
        'soc_prev'           : soc,
        'energy_norm_prev'   : energy_norm,
        'battery_temp'       : new_bat_temp,

        # Cumulatives (net energy — energy_wh is negative during regen)
        'cum_wh'             : s['cum_wh'] + energy_wh,
        'cum_km'             : s['cum_km'] + velocity / 3.6 * DT / 1000,

        # LSTM buffers
        'motor_seq_buffer'   : s['motor_seq_buffer'],
        'dyn_seq_buffer'     : s['dyn_seq_buffer'],
        'rng_seq_buffer'     : s['rng_seq_buffer'],
        '_motor_prefill'     : s.get('_motor_prefill', False),
        '_rng_prefill'       : s.get('_rng_prefill', False),

        # Display outputs
        'u_d'  : u_d,       'u_q'  : u_q,
        'i_d'  : i_d,       'i_q'  : i_q,
        'torque': float(abs(torque)) if regen == 0 else torque,   'rpm'   : rpm,
        'pm_temp'    : pm_temp,
        'stator_temp': stator_t,
        'velocity'   : velocity,
        'accel'      : accel,
        'soc'        : soc,
        'energy_norm': energy_norm,
        'remaining_km': remaining_km,
        'duty_cycle' : duty,
        'v_phase'    : v_phase,
        'slope'      : slope,
        'regen'      : regen,
        'throttle'   : throttle,
        'battery_temp_display': new_bat_temp,
    }


# ─────────────────────────────────────────────
#  ALERT GENERATION
# ─────────────────────────────────────────────
def compute_alerts(state):
    """Returns list of (type, message) tuples based on current state."""
    alerts = []

    if state['pm_temp'] >= TEMP_CRITICAL:
        alerts.append(('danger',
            f"CRITICAL: Rotor temp {state['pm_temp']:.1f}C "
            f"— reduce throttle immediately"))
    elif state['pm_temp'] >= TEMP_WARN:
        alerts.append(('warning',
            f"WARNING: Rotor temp {state['pm_temp']:.1f}C "
            f"— approaching thermal limit"))

    if state['stator_temp'] >= TEMP_CRITICAL:
        alerts.append(('danger',
            f"CRITICAL: Stator temp {state['stator_temp']:.1f}C"))
    elif state['stator_temp'] >= TEMP_WARN:
        alerts.append(('warning',
            f"WARNING: Stator temp {state['stator_temp']:.1f}C"))

    if state['soc'] <= SOC_CRITICAL:
        alerts.append(('danger',
            f"BATTERY CRITICAL: {state['soc']:.1f}% "
            f"— {state['remaining_km']:.1f} km left"))
    elif state['soc'] <= SOC_WARN:
        alerts.append(('warning',
            f"LOW BATTERY: {state['soc']:.1f}% "
            f"— {state['remaining_km']:.1f} km remaining"))

    if not alerts:
        alerts.append(('ok',
            f"All systems nominal "
            f"— {state['remaining_km']:.1f} km range remaining"))

    return alerts
