"""
state.py — Session state initialisation and history management
"""

HISTORY_LEN = 300   # max steps kept in live chart history

# All signal keys tracked in history
HISTORY_KEYS = [
    'step', 'throttle', 'slope', 'regen',
    'u_d', 'u_q', 'v_phase', 'duty_cycle',
    'i_d', 'i_q', 'torque', 'rpm',
    'pm_temp', 'stator_temp',
    'velocity', 'accel',
    'soc', 'remaining_km',
    'energy_norm', 'battery_temp_display',
]


def init_history():
    """Returns empty history dictionary."""
    return {k: [] for k in HISTORY_KEYS}


def append_history(history, step, state):
    """
    Appends current state values to rolling history.
    Trims to HISTORY_LEN to keep memory bounded.
    """
    history['step'].append(step)
    for k in HISTORY_KEYS:
        if k == 'step':
            continue
        history[k].append(state.get(k, 0.0))

    # Trim to last HISTORY_LEN points
    if len(history['step']) > HISTORY_LEN:
        for k in history:
            history[k] = history[k][-HISTORY_LEN:]

    return history
