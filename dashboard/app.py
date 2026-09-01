"""
E-Scooter Digital Shadow Dashboard
====================================
Reads live state from sensor_broker.py via shared_state.json.
Does NOT run ML models — broker handles that.
Does NOT send to Unity — broker handles that.

Run order:
    1. python sensor_broker.py your_scenario.csv
    2. streamlit run app.py

Required folder structure:
    Models/           <- trained model files
    shared_state.json <- written by sensor_broker.py (auto-created)
    styles.py
    models.py         <- only used for constants, not for inference here
    charts.py
    state.py
    app.py            <- this file
    sensor_broker.py  <- runs models, feeds Unity + this dashboard
"""

import io
import json
import time
import zipfile

import numpy as np
import pandas as pd
import streamlit as st

from styles import inject_css
from models import (
    TEMP_WARN, TEMP_CRITICAL,
    SOC_WARN, SOC_CRITICAL,
    SCOOTER_MASS, SCOOTER_BATTERY_WH,
)
from charts import (
    make_gauge, make_soc_bar, metric_card,
    make_speed_throttle_regen_chart,
    make_battery_consumption_chart,
    make_regen_detail_chart,
    make_range_estimation_chart,
)
from state import init_history, append_history

# ─────────────────────────────────────────────
#  SHARED STATE FILE PATH
#  Must match the path in sensor_broker.py
# ─────────────────────────────────────────────
STATE_FILE = "shared_state.json"

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="E-Scooter Digital Shadow",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ─────────────────────────────────────────────
#  READ SHARED STATE FROM BROKER
# ─────────────────────────────────────────────
def read_broker_state():
    """
    Reads the latest state written by sensor_broker.py.
    Returns None if file doesn't exist or broker hasn't started yet.
    """
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return None


def broker_is_running():
    """Check if broker has written a recent state (within last 5 seconds)."""
    state = read_broker_state()
    if state is None:
        return False
    ts = state.get("timestamp", 0)
    return (time.time() - ts) < 5.0


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="dt-header">
    <div style="display:flex;justify-content:space-between;align-items:center;">
        <div>
            <p class="dt-title">E-Scooter Digital Shadow</p>
            <p class="dt-subtitle">
                LIVE MONITORING — BROKER-DRIVEN — UNITY + DASHBOARD IN SYNC
            </p>
        </div>
        <div style="text-align:right;">
            <span class="dt-status">DASHBOARD ONLINE</span>
            <p style="color:#8899aa;font-size:0.75rem;margin:4px 0 0 0;letter-spacing:1px;">
                READS FROM BROKER
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR — broker instructions + model info
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <p style="font-family:Rajdhani;font-size:1.1rem;color:#00d4ff;
              letter-spacing:2px;text-transform:uppercase;
              border-bottom:1px solid #2a3a4a;padding-bottom:8px;">
    How To Run
    </p>""", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem;color:#8899aa;
                letter-spacing:1px;line-height:2.2;">
    <b style="color:#00ff88;">Step 1</b> — Open Unity → Press Play<br>
    <b style="color:#00ff88;">Step 2</b> — Run the broker:<br>
    <code style="color:#00d4ff;font-size:0.72rem;">
    python sensor_broker.py your_file.csv
    </code><br>
    <b style="color:#00ff88;">Step 3</b> — This dashboard auto-updates<br><br>
    <div style="color:#ff6b35;">
    Do NOT run simulation from this page.<br>
    Broker controls everything.
    </div>
    </div>
    <hr style="border-color:#2a3a4a;margin:16px 0;">
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="font-family:Rajdhani;font-size:1.1rem;color:#00d4ff;
              letter-spacing:2px;text-transform:uppercase;
              border-bottom:1px solid #2a3a4a;padding-bottom:8px;">
    Model Pipeline
    </p>""", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem;color:#8899aa;
                letter-spacing:1px;line-height:2.0;">
    <b style="color:#e8f0fe;">Controller</b> — XGBoost<br>
    &nbsp;&nbsp;throttle → u_d, u_q<br><br>
    <b style="color:#e8f0fe;">Motor</b> — LSTM (seq=50)<br>
    &nbsp;&nbsp;u_d, u_q → i, torque, RPM, T<br><br>
    <b style="color:#e8f0fe;">Dynamics</b> — LSTM (seq=150)<br>
    &nbsp;&nbsp;15s history → velocity, accel<br><br>
    <b style="color:#e8f0fe;">Range</b> — LSTM (seq=200)<br>
    &nbsp;&nbsp;20s history → SoC, remaining km<br><br>
    <div style="color:#00ff88;">
    All models run in sensor_broker.py<br>
    Dashboard only reads results.
    </div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  BROKER STATUS CHECK
# ─────────────────────────────────────────────
broker_state = read_broker_state()

if broker_state is None:
    st.markdown("""
    <div style="background:#1a2235;border:1px solid #ff6b35;border-radius:12px;
                padding:50px 40px;text-align:center;margin-top:30px;">
        <div style="font-size:2.5rem;margin-bottom:20px;">⏳</div>
        <div style="font-family:Rajdhani;font-size:1.3rem;color:#ff6b35;
                    letter-spacing:2px;text-transform:uppercase;">
            Waiting for sensor_broker.py to start
        </div>
        <div style="color:#8899aa;font-size:0.9rem;margin-top:14px;line-height:2.2;">
            Open a terminal and run:<br>
            <code style="color:#00d4ff;">
                python sensor_broker.py your_scenario.csv
            </code><br><br>
            This dashboard will automatically update once the broker starts.
        </div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(1)
    st.rerun()

# ─────────────────────────────────────────────
#  SCENARIO INFO BAR (from broker state)
# ─────────────────────────────────────────────
scenario_name = broker_state.get("scenario_name", "Unknown Scenario")
total_steps   = broker_state.get("total_steps", 0)
soc_init      = broker_state.get("soc_init", 80.0)
current_step  = broker_state.get("current_step", 0)
sim_complete  = broker_state.get("sim_complete", False)

i1, i2, i3, i4, i5 = st.columns(5)
with i1:
    st.markdown(metric_card("SCENARIO",
                scenario_name[:14], "", "Live"), unsafe_allow_html=True)
with i2:
    st.markdown(metric_card("PROGRESS",
                f"{current_step:,}", "steps",
                f"of {total_steps:,} total"), unsafe_allow_html=True)
with i3:
    st.markdown(metric_card("STARTING SOC",
                f"{soc_init:.0f}", "%", "From broker"),
                unsafe_allow_html=True)
with i4:
    elapsed = broker_state.get("elapsed_s", current_step * 0.1)
    st.markdown(metric_card("ELAPSED",
                f"{elapsed:.1f}", "s",
                f"{elapsed/60:.1f} min"), unsafe_allow_html=True)
with i5:
    data_src = broker_state.get("data_source", "csv")
    st.markdown(metric_card("DATA SOURCE",
                data_src.upper(), "",
                "CSV / ESP32"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PROGRESS BAR
# ─────────────────────────────────────────────
if total_steps > 0:
    pct = min(int(current_step / total_steps * 100), 100)
    if sim_complete:
        st.progress(100, text="Simulation complete — showing final state")
    else:
        st.progress(pct,
            text=f"Running... {current_step:,} / {total_steps:,} steps ({pct}%)")
else:
    st.progress(0, text="Waiting for broker...")

# ─────────────────────────────────────────────
#  ALERT BANNERS
# ─────────────────────────────────────────────
pm_temp  = broker_state.get("motor_temp", 25.0)
soc      = broker_state.get("soc", 100.0)
rem_km   = broker_state.get("remaining_range", 35.0)

ahtml = ""
if pm_temp >= TEMP_CRITICAL:
    ahtml += (f'<div class="alert-banner">CRITICAL: Rotor temp {pm_temp:.1f}C '
              f'— reduce throttle immediately</div>')
elif pm_temp >= TEMP_WARN:
    ahtml += (f'<div class="warn-banner">WARNING: Rotor temp {pm_temp:.1f}C '
              f'— approaching thermal limit</div>')

if soc <= SOC_CRITICAL:
    ahtml += (f'<div class="alert-banner">BATTERY CRITICAL: {soc:.1f}% '
              f'— {rem_km:.1f} km left</div>')
elif soc <= SOC_WARN:
    ahtml += (f'<div class="warn-banner">LOW BATTERY: {soc:.1f}% '
              f'— {rem_km:.1f} km remaining</div>')

if not ahtml:
    ahtml = (f'<div class="ok-banner">All systems nominal '
             f'— {rem_km:.1f} km range remaining</div>')

st.markdown(ahtml, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LIVE METRIC CARDS
# ─────────────────────────────────────────────
speed   = broker_state.get("speed", 0.0)
rpm     = broker_state.get("rpm", 0.0)
torque  = broker_state.get("torque", 0.0)
accel   = broker_state.get("accel", 0.0)
stator  = broker_state.get("stator_temp", 25.0)
v_phase = broker_state.get("v_phase", 0.0)
u_d     = broker_state.get("u_d", 0.0)
u_q     = broker_state.get("u_q", 0.0)
duty    = broker_state.get("duty_cycle", 0.0)
i_q     = broker_state.get("i_q", 0.0)

pm_s  = ("danger"  if pm_temp >= TEMP_CRITICAL else
         "warning" if pm_temp >= TEMP_WARN     else "normal")
soc_s = ("danger"  if soc <= SOC_CRITICAL else
         "warning" if soc <= SOC_WARN      else "good")

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: st.markdown(metric_card("VEHICLE SPEED",
                      f"{speed:.1f}", "km/h",
                      f"Accel: {accel:+.2f} m/s2"),
                      unsafe_allow_html=True)
with c2: st.markdown(metric_card("MOTOR RPM",
                      f"{rpm:.0f}", "RPM",
                      f"Torque: {torque:.1f} Nm"),
                      unsafe_allow_html=True)
with c3: st.markdown(metric_card("ROTOR TEMP",
                      f"{pm_temp:.1f}", "C",
                      f"Stator: {stator:.1f}C",
                      status=pm_s), unsafe_allow_html=True)
with c4: st.markdown(metric_card("BATTERY SOC",
                      f"{soc:.1f}", "%",
                      f"Range: {rem_km:.1f} km",
                      status=soc_s), unsafe_allow_html=True)
with c5: st.markdown(metric_card("PHASE VOLTAGE",
                      f"{v_phase:.1f}", "V",
                      f"u_d:{u_d:.1f} u_q:{u_q:.1f}"),
                      unsafe_allow_html=True)
with c6: st.markdown(metric_card("DUTY CYCLE",
                      f"{duty*100:.1f}", "%",
                      f"I_q: {i_q:.1f} A"),
                      unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  GAUGES
# ─────────────────────────────────────────────
g1, g2, g3, g4, g5 = st.columns(5)
with g1: st.plotly_chart(
    make_gauge(speed, "SPEED", 0, 80, "km/h"),
    use_container_width=True, key="g_spd")
with g2: st.plotly_chart(
    make_gauge(rpm, "MOTOR RPM", 0, 6000, "RPM"),
    use_container_width=True, key="g_rpm")
with g3: st.plotly_chart(
    make_gauge(pm_temp, "ROTOR TEMP", 0, 130, "C",
               warn=TEMP_WARN, critical=TEMP_CRITICAL,
               color="#ff6b35"),
    use_container_width=True, key="g_tmp")
with g4: st.plotly_chart(
    make_gauge(soc, "BATTERY SOC", 0, 100, "%",
               warn=SOC_WARN, critical=SOC_CRITICAL,
               color="#00ff88"),
    use_container_width=True, key="g_soc")
with g5: st.plotly_chart(
    make_gauge(abs(torque), "TORQUE", 0, 250, "Nm",
               color="#ffd700"),
    use_container_width=True, key="g_trq")

# ─────────────────────────────────────────────
#  SOC BAR + SECONDARY METRICS
# ─────────────────────────────────────────────
rb1, rb2, rb3 = st.columns([3, 1, 1])
with rb1:
    st.plotly_chart(make_soc_bar(soc),
                    use_container_width=True, key="soc_bar")
with rb2:
    st.markdown(metric_card("REMAINING RANGE",
                 f"{rem_km:.1f}", "km",
                 f"At {speed:.0f} km/h"),
                 unsafe_allow_html=True)
with rb3:
    energy_norm = broker_state.get("energy_norm", 0.0)
    e_mwh = abs(energy_norm) * SCOOTER_MASS * 1000
    st.markdown(metric_card("ENERGY/STEP",
                 f"{e_mwh:.2f}", "mWh",
                 "Scooter normalised"),
                 unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  LIVE HISTORY CHARTS
#  Built from broker's rolling history in shared_state.json
# ─────────────────────────────────────────────
st.markdown('<p class="section-header">LIVE GRAPHS</p>',
            unsafe_allow_html=True)

history = broker_state.get("history", None)

if history and len(history.get("step", [])) > 5:
    st.plotly_chart(
        make_speed_throttle_regen_chart(history),
        use_container_width=True, key="lg1")
    st.plotly_chart(
        make_battery_consumption_chart(history),
        use_container_width=True, key="lg2")
    st.plotly_chart(
        make_regen_detail_chart(history),
        use_container_width=True, key="lg3")
    st.plotly_chart(
        make_range_estimation_chart(history),
        use_container_width=True, key="lg4")
else:
    st.info("Graphs will appear after a few seconds of simulation data...")

# ─────────────────────────────────────────────
#  FINAL SUMMARY (only shown when sim_complete)
# ─────────────────────────────────────────────
if sim_complete:
    summary = broker_state.get("summary", None)
    full_history = broker_state.get("full_history", None)

    if summary and full_history:
        s = summary
        regen_pct = (s.get("regen_recovered_wh", 0) /
                     max(s.get("energy_consumed_wh", 0.001), 0.001) * 100)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#1a2235;border:1px solid #00d4ff;
                    border-radius:12px;padding:20px 28px;margin-bottom:20px;
                    position:relative;overflow:hidden;">
            <div style="position:absolute;top:0;left:0;right:0;height:2px;
                background:linear-gradient(90deg,transparent,
                #00d4ff,#00ff88,transparent);"></div>
            <div style="font-family:Rajdhani;font-size:1.4rem;font-weight:700;
                        color:#00d4ff;letter-spacing:2px;text-transform:uppercase;">
                TRIP COMPLETE — {s.get("scenario_name", "")}
            </div>
            <div style="color:#8899aa;font-size:0.8rem;margin-top:4px;letter-spacing:1px;">
                {s.get("total_steps",0):,} steps &nbsp;|&nbsp;
                {s.get("duration_s",0):.0f}s &nbsp;|&nbsp;
                {s.get("duration_s",0)/60:.1f} minutes simulated
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-header">TRIP SUMMARY</p>',
                    unsafe_allow_html=True)

        sm1, sm2, sm3, sm4, sm5, sm6 = st.columns(6)
        with sm1: st.markdown(metric_card("DISTANCE",
                               f"{s.get('total_distance_km',0):.2f}", "km",
                               f"Avg: {s.get('avg_speed_kmh',0):.1f} km/h"),
                               unsafe_allow_html=True)
        with sm2: st.markdown(metric_card("SOC USED",
                               f"{s.get('soc_consumed_pct',0):.1f}", "%",
                               f"Final: {s.get('soc_final',0):.1f}%",
                               status="warning" if s.get("soc_consumed_pct",0) > 50 else "normal"),
                               unsafe_allow_html=True)
        with sm3: st.markdown(metric_card("ENERGY USED",
                               f"{s.get('energy_consumed_wh',0):.1f}", "Wh",
                               f"Eff: {s.get('avg_efficiency_wh_km',0):.1f} Wh/km"),
                               unsafe_allow_html=True)
        with sm4: st.markdown(metric_card("REGEN RECOVERED",
                               f"{s.get('regen_recovered_wh',0):.2f}", "Wh",
                               f"Net: {s.get('net_energy_wh',0):.1f} Wh",
                               status="good"), unsafe_allow_html=True)
        with sm5:
            pk = s.get("peak_rotor_temp", 0)
            tp_s = ("danger"  if pk >= TEMP_CRITICAL else
                    "warning" if pk >= TEMP_WARN     else "normal")
            st.markdown(metric_card("PEAK ROTOR TEMP",
                         f"{pk:.1f}", "C",
                         f">{TEMP_WARN:.0f}C: {s.get('time_above_warn_s',0):.0f}s",
                         status=tp_s), unsafe_allow_html=True)
        with sm6: st.markdown(metric_card("FINAL RANGE",
                               f"{s.get('final_range_km',0):.1f}", "km",
                               "Estimated remaining",
                               status=("danger"  if s.get("final_range_km",0) < 5  else
                                       "warning" if s.get("final_range_km",0) < 10 else "good")),
                               unsafe_allow_html=True)

        # ── EXPORT ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">EXPORT RESULTS</p>',
                    unsafe_allow_html=True)

        # Build export ZIP from broker's full_history and summary
        zip_buffer = io.BytesIO()
        h = full_history

        df_sim = pd.DataFrame({
            "step":            h.get("step", []),
            "speed_kmh":       h.get("velocity", []),
            "rpm":             h.get("rpm", []),
            "motor_temp_C":    h.get("pm_temp", []),
            "soc_pct":         h.get("soc", []),
            "remaining_km":    h.get("remaining_km", []),
            "torque_Nm":       h.get("torque", []),
            "throttle_pct":    h.get("throttle", []),
            "slope":           h.get("slope", []),
            "regen":           h.get("regen", []),
            "u_d":             h.get("u_d", []),
            "u_q":             h.get("u_q", []),
            "i_d":             h.get("i_d", []),
            "i_q":             h.get("i_q", []),
            "accel_ms2":       h.get("accel", []),
            "energy_norm":     h.get("energy_norm", []),
            "duty_cycle":      h.get("duty_cycle", []),
            "v_phase":         h.get("v_phase", []),
            "stator_temp_C":   h.get("stator_temp", []),
            "battery_temp_C":  h.get("battery_temp_display", []),
        })

        df_summary = pd.DataFrame([{
            "scenario":               s.get("scenario_name", ""),
            "total_steps":            s.get("total_steps", 0),
            "duration_s":             s.get("duration_s", 0),
            "total_distance_km":      s.get("total_distance_km", 0),
            "avg_speed_kmh":          s.get("avg_speed_kmh", 0),
            "starting_soc_pct":       s.get("soc_init", 80),
            "final_soc_pct":          s.get("soc_final", 0),
            "soc_consumed_pct":       s.get("soc_consumed_pct", 0),
            "energy_consumed_wh":     s.get("energy_consumed_wh", 0),
            "regen_recovered_wh":     s.get("regen_recovered_wh", 0),
            "net_energy_wh":          s.get("net_energy_wh", 0),
            "regen_recovery_pct":     round(regen_pct, 2),
            "avg_efficiency_wh_per_km": s.get("avg_efficiency_wh_km", 0),
            "peak_rotor_temp_C":      s.get("peak_rotor_temp", 0),
            "time_above_warn_80C_s":  s.get("time_above_warn_s", 0),
            "time_above_critical_s":  s.get("time_critical_s", 0),
            "final_range_km":         s.get("final_range_km", 0),
        }])

        safe_name = scenario_name.replace(" ", "_").lower()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{safe_name}_simulation_data.csv",
                        df_sim.to_csv(index=False).encode("utf-8"))
            zf.writestr(f"{safe_name}_summary_metrics.csv",
                        df_summary.to_csv(index=False).encode("utf-8"))

        zip_buffer.seek(0)
        ec1, ec2, ec3 = st.columns([1, 2, 1])
        with ec2:
            st.download_button(
                label="⬇  Download All Results (ZIP)",
                data=zip_buffer.read(),
                file_name=f"{safe_name}_digital_shadow_results.zip",
                mime="application/zip",
                use_container_width=True,
                type="primary",
            )

        # Full trip graphs from broker's full_history
        st.markdown('<p class="section-header">FULL TRIP GRAPHS</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(make_speed_throttle_regen_chart(full_history),
                        use_container_width=True, key="fs_g1")
        st.plotly_chart(make_battery_consumption_chart(full_history),
                        use_container_width=True, key="fs_g2")
        st.plotly_chart(make_regen_detail_chart(full_history),
                        use_container_width=True, key="fs_g3")
        st.plotly_chart(make_range_estimation_chart(full_history),
                        use_container_width=True, key="fs_g4")

# ─────────────────────────────────────────────
#  AUTO-REFRESH every 1 second (matches broker sync rate)
#  Only refresh while simulation is running
# ─────────────────────────────────────────────
if not sim_complete:
    time.sleep(1.0)
    st.rerun()
