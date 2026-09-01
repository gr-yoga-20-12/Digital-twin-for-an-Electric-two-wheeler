"""
charts.py — All chart, gauge, and visualisation functions
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────
#  SHARED LAYOUT DEFAULTS
# ─────────────────────────────────────────────
_TRANSPARENT = 'rgba(0,0,0,0)'
_GRID_COLOR  = '#1a2235'
_TICK_COLOR  = '#8899aa'
_TICK_FONT   = dict(size=8, color=_TICK_COLOR)
_TITLE_FONT  = dict(size=10, color=_TICK_COLOR, family='Rajdhani')


def _base_layout(height=160, margin=None):
    m = margin or dict(l=40, r=10, t=25, b=30)
    return dict(
        height=height,
        margin=m,
        paper_bgcolor=_TRANSPARENT,
        plot_bgcolor=_TRANSPARENT,
        showlegend=False,
        font_color='#e8f0fe',
    )


# ─────────────────────────────────────────────
#  GAUGE
# ─────────────────────────────────────────────
def make_gauge(value, title, min_val, max_val, unit,
               warn=None, critical=None, color='#00d4ff'):
    """Animated semicircle gauge with optional warn/critical thresholds."""
    if critical and value >= critical:
        bar_color = '#ff3366'
    elif warn and value >= warn:
        bar_color = '#ff6b35'
    else:
        bar_color = color

    steps = [{'range': [min_val, max_val], 'color': '#1a2235'}]
    threshold = None
    if warn:
        steps = [
            {'range': [min_val, warn], 'color': '#1a2235'},
            {'range': [warn, critical or max_val], 'color': '#2a1a1a'},
        ]
        if critical:
            steps.append({'range': [critical, max_val], 'color': '#3a0a0a'})
        threshold = {
            'line': {'color': '#ff3366', 'width': 2},
            'thickness': 0.75,
            'value': critical or warn,
        }

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': f' {unit}',
                'font': {'size': 18, 'color': '#e8f0fe',
                         'family': 'JetBrains Mono'}},
        title={'text': title,
               'font': {'size': 11, 'color': '#8899aa', 'family': 'Rajdhani'}},
        gauge={
            'axis': {'range': [min_val, max_val],
                     'tickcolor': '#8899aa',
                     'tickfont': {'size': 9, 'color': '#8899aa'},
                     'nticks': 5},
            'bar': {'color': bar_color, 'thickness': 0.25},
            'bgcolor': _TRANSPARENT,
            'bordercolor': '#2a3a4a',
            'steps': steps,
            'threshold': threshold,
        }
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor=_TRANSPARENT,
        plot_bgcolor=_TRANSPARENT,
        font_color='#e8f0fe',
    )
    return fig


# ─────────────────────────────────────────────
#  SIMPLE LINE CHART
# ─────────────────────────────────────────────
def make_line(history, key, title, unit, color='#00d4ff',
              warn_line=None, crit_line=None, fill=True):
    """Single signal line chart for live history panels."""
    fig = go.Figure()
    x, y = history['step'], history[key]

    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines',
        line=dict(color=color, width=2),
        fill='tozeroy' if fill else 'none',
        fillcolor=color + '14' if fill else None,
        name=title,
    ))
    if warn_line:
        fig.add_hline(y=warn_line, line_dash='dash', line_color='#ff6b35',
                      line_width=1, annotation_text='WARN',
                      annotation_font_color='#ff6b35',
                      annotation_font_size=9)
    if crit_line:
        fig.add_hline(y=crit_line, line_dash='dash', line_color='#ff3366',
                      line_width=1, annotation_text='CRIT',
                      annotation_font_color='#ff3366',
                      annotation_font_size=9)

    fig.update_layout(
        **_base_layout(),
        title=dict(text=f'{title} ({unit})', font=_TITLE_FONT),
        xaxis=dict(showgrid=False, color=_TICK_COLOR,
                   tickfont=_TICK_FONT, title='Step'),
        yaxis=dict(showgrid=True, gridcolor=_GRID_COLOR,
                   color=_TICK_COLOR, tickfont=_TICK_FONT),
    )
    return fig


# ─────────────────────────────────────────────
#  SOC PROGRESS BAR
# ─────────────────────────────────────────────
def make_soc_bar(soc, soc_warn=25.0, soc_critical=10.0):
    """Horizontal battery SoC progress bar."""
    color = ('#ff3366' if soc < soc_critical else
             '#ff6b35' if soc < soc_warn     else
             '#00d4ff' if soc < 60           else '#00ff88')
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[soc], y=[''],
        orientation='h',
        marker_color=color, marker_line_width=0,
        text=[f'{soc:.1f}%'], textposition='inside',
        textfont=dict(size=14, color='white', family='JetBrains Mono'),
    ))
    fig.add_trace(go.Bar(
        x=[100 - soc], y=[''],
        orientation='h',
        marker_color='#1a2235', marker_line_width=0,
        showlegend=False,
    ))
    fig.update_layout(
        height=55, barmode='stack',
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=_TRANSPARENT, plot_bgcolor=_TRANSPARENT,
        showlegend=False,
        xaxis=dict(range=[0, 100], visible=False),
        yaxis=dict(visible=False),
    )
    return fig


# ─────────────────────────────────────────────
#  METRIC CARD HTML
# ─────────────────────────────────────────────
def metric_card(label, value, unit, sub='', status='normal'):
    """Returns HTML string for a styled metric card."""
    cls = {'normal': '', 'warning': 'warning',
           'danger': 'danger', 'good': 'good'}
    return f"""
    <div class="metric-card {cls.get(status, '')}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}<span class="metric-unit">{unit}</span></div>
        <div class="metric-sub">{sub}</div>
    </div>"""


# ─────────────────────────────────────────────
#  FLOW DIAGRAM HTML
# ─────────────────────────────────────────────
def flow_diagram_html():
    """Returns HTML for the 4-model feedback loop diagram."""
    box = lambda title, algo, io: f"""
    <div class="flow-box active">
        <div style="color:#00d4ff;font-size:0.9rem;font-weight:700;">{title}</div>
        <div style="color:#8899aa;font-size:0.7rem;margin-top:2px;">{algo}</div>
        <div style="color:#e8f0fe;font-size:0.72rem;margin-top:6px;">{io}</div>
    </div>"""
    arrow = '<div style="text-align:center;color:#00d4ff;font-size:1.5rem;margin-top:20px;">→</div>'
    feedback = """
    <div style="text-align:center;margin-top:8px;color:#8899aa;
                font-size:0.75rem;letter-spacing:1px;">
        Feedback: i_d, i_q, RPM, T_pm → Controller next step &nbsp;|&nbsp;
        v, torque, SoC → Range next step
    </div>"""
    return (box("CONTROLLER", "XGBoost",    "throttle → u_d, u_q"),
            arrow,
            box("MOTOR",      "LSTM (x50)", "u_d,u_q → i,torque,RPM,T"),
            arrow,
            box("DYNAMICS",   "XGBoost",    "throttle,slope → v,a"),
            arrow,
            box("RANGE",      "XGBoost",    "v,torque,T → SoC, km"),
            feedback)


# ══════════════════════════════════════════════
#  CSV PLAYER SPECIFIC CHARTS
# ══════════════════════════════════════════════

def make_speed_throttle_regen_chart(history):
    """
    Combined chart — Graph 1 for CSV player summary.
    Throttle as filled area (background), Speed as cyan line,
    Regen as green spikes on right Y-axis.
    """
    steps = history['step']
    fig   = make_subplots(specs=[[{"secondary_y": True}]])

    # ── Throttle — filled orange area (input / cause) ──
    fig.add_trace(go.Scatter(
        x=steps, y=history['throttle'],
        mode='lines',
        name='Throttle (%)',
        line=dict(color='#ff6b35', width=1),
        fill='tozeroy',
        fillcolor='rgba(255,107,53,0.20)',
    ), secondary_y=False)

    # ── Speed — solid cyan line (response) ──
    fig.add_trace(go.Scatter(
        x=steps, y=history['velocity'],
        mode='lines',
        name='Speed (km/h)',
        line=dict(color='#00ff88', width=2.5),
    ), secondary_y=False)

    # ── Regen — green spike bars on right axis ──
    fig.add_trace(go.Scatter(
        x=steps, y=history['regen'],
        mode='lines',
        name='Regen',
        line=dict(color='#00d4ff', width=1.5, dash='dot'),
        fill='tozeroy',
        fillcolor='rgba(0,212,255,0.15)',
    ), secondary_y=True)

    fig.update_layout(
        height=280,
        margin=dict(l=50, r=60, t=35, b=40),
        paper_bgcolor=_TRANSPARENT,
        plot_bgcolor=_TRANSPARENT,
        legend=dict(orientation='h', y=1.08, font=dict(size=9, color='#8899aa')),
        title=dict(text='Speed / Throttle / Regen Braking',
                   font=dict(size=11, color='#00d4ff', family='Rajdhani')),
    )
    fig.update_yaxes(
        title_text='km/h  |  %',
        gridcolor=_GRID_COLOR, color=_TICK_COLOR,
        tickfont=_TICK_FONT, secondary_y=False,
    )
    fig.update_yaxes(
        title_text='Regen (0–0.5)',
        range=[0, 1.2], color='#00d4ff',
        tickfont=dict(size=8, color='#00d4ff'),
        secondary_y=True,
    )
    fig.update_xaxes(
        showgrid=False, color=_TICK_COLOR,
        tickfont=_TICK_FONT, title='Step',
    )
    return fig


def make_battery_consumption_chart(history):
    """
    Combined chart — Graph 2.
    SoC as left Y-axis line, Energy per step as right Y-axis line.
    """
    steps = history['step']

    # Convert energy_norm (Wh/kg) to Wh using scooter mass
    energy_wh = [abs(e) * 90.0 * 1000 for e in history['energy_norm']]  # mWh

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # ── SoC — cyan filled area ──
    fig.add_trace(go.Scatter(
        x=steps, y=history['soc'],
        mode='lines',
        name='Battery SoC (%)',
        line=dict(color='#00d4ff', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(0,212,255,0.12)',
    ), secondary_y=False)

    # ── Energy per step — orange line on right axis ──
    fig.add_trace(go.Scatter(
        x=steps, y=energy_wh,
        mode='lines',
        name='Energy/step (mWh)',
        line=dict(color='#ff6b35', width=1.5),
    ), secondary_y=True)

    fig.update_layout(
        height=280,
        margin=dict(l=50, r=60, t=35, b=40),
        paper_bgcolor=_TRANSPARENT,
        plot_bgcolor=_TRANSPARENT,
        legend=dict(orientation='h', y=1.08, font=dict(size=9, color='#8899aa')),
        title=dict(text='Battery SoC & Energy Consumption',
                   font=dict(size=11, color='#00d4ff', family='Rajdhani')),
    )
    fig.update_yaxes(
        title_text='SoC (%)',
        range=[0, 105],
        gridcolor=_GRID_COLOR, color=_TICK_COLOR,
        tickfont=_TICK_FONT, secondary_y=False,
    )
    fig.update_yaxes(
        title_text='Energy/step (mWh)',
        color='#ff6b35',
        tickfont=dict(size=8, color='#ff6b35'),
        secondary_y=True,
    )
    fig.update_xaxes(
        showgrid=False, color=_TICK_COLOR,
        tickfont=_TICK_FONT, title='Step',
    )
    return fig


def make_regen_detail_chart(history):
    """
    Graph 3 — Regenerative braking detail.
    Regen signal, i_q current (negative during regen),
    and cumulative energy recovered.
    """
    steps = history['step']

    # Cumulative regen energy recovered
    regen_energy_mwh = []
    cum = 0.0
    for i, (r, e) in enumerate(zip(history['regen'], history['energy_norm'])):
        if r > 0 and e < 0:
            cum += abs(e) * 90.0 * 1000   # mWh recovered
        regen_energy_mwh.append(cum)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # ── Regen signal — filled green area ──
    fig.add_trace(go.Scatter(
        x=steps, y=[r * 100 for r in history['regen']],
        mode='lines',
        name='Regen Signal (%)',
        line=dict(color='#00ff88', width=1),
        fill='tozeroy',
        fillcolor='rgba(0,255,136,0.15)',
    ), secondary_y=False)

    # ── i_q current — yellow line (negative = regen) ──
    fig.add_trace(go.Scatter(
        x=steps, y=history['i_q'],
        mode='lines',
        name='i_q Current (A)',
        line=dict(color='#ffd700', width=1.5),
    ), secondary_y=False)

    # ── Cumulative energy recovered — right axis ──
    fig.add_trace(go.Scatter(
        x=steps, y=regen_energy_mwh,
        mode='lines',
        name='Cumulative Regen (mWh)',
        line=dict(color='#88ffdd', width=2, dash='dash'),
    ), secondary_y=True)

    fig.update_layout(
        height=280,
        margin=dict(l=50, r=60, t=35, b=40),
        paper_bgcolor=_TRANSPARENT,
        plot_bgcolor=_TRANSPARENT,
        legend=dict(orientation='h', y=1.08, font=dict(size=9, color='#8899aa')),
        title=dict(text='Regenerative Braking Detail',
                   font=dict(size=11, color='#00d4ff', family='Rajdhani')),
    )
    fig.update_yaxes(
        title_text='Regen (%) | i_q (A)',
        gridcolor=_GRID_COLOR, color=_TICK_COLOR,
        tickfont=_TICK_FONT, secondary_y=False,
    )
    fig.update_yaxes(
        title_text='Cumulative Regen (mWh)',
        color='#88ffdd',
        tickfont=dict(size=8, color='#88ffdd'),
        secondary_y=True,
    )
    fig.update_xaxes(
        showgrid=False, color=_TICK_COLOR,
        tickfont=_TICK_FONT, title='Step',
    )
    return fig


def make_range_estimation_chart(history):
    """
    Graph 4 — Range estimation breakdown.
    Remaining range (left), SoC and slope and energy rate (right).
    Shows WHY range changes — driven by slope and energy rate.
    """
    steps    = history['step']
    slope_pct = [s * 100 for s in history['slope']]

    # Energy rate in Wh/km — only meaningful when moving
    energy_wh_per_km = []
    for v, e in zip(history['velocity'], history['energy_norm']):
        if v > 1.0:
            steps_per_km = 1000 / (v / 3.6 * 0.1)
            wh_per_km    = abs(e) * 90.0 * steps_per_km
            energy_wh_per_km.append(float(np.clip(wh_per_km, 0, 80)))
        else:
            energy_wh_per_km.append(0.0)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # ── Remaining range — primary cyan line ──
    fig.add_trace(go.Scatter(
        x=steps, y=history['remaining_km'],
        mode='lines',
        name='Range Left (km)',
        line=dict(color='#44ddff', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(68,221,255,0.10)',
    ), secondary_y=False)

    # ── Slope — yellow shaded area on right axis ──
    fig.add_trace(go.Scatter(
        x=steps, y=slope_pct,
        mode='lines',
        name='Slope (%)',
        line=dict(color='#ffd700', width=1),
        fill='tozeroy',
        fillcolor='rgba(255,215,0,0.10)',
    ), secondary_y=True)

    # ── Energy rate — orange dotted line on right axis ──
    fig.add_trace(go.Scatter(
        x=steps, y=energy_wh_per_km,
        mode='lines',
        name='Energy Rate (Wh/km)',
        line=dict(color='#ff6b35', width=1.5, dash='dot'),
    ), secondary_y=True)

    fig.update_layout(
        height=280,
        margin=dict(l=50, r=70, t=35, b=40),
        paper_bgcolor=_TRANSPARENT,
        plot_bgcolor=_TRANSPARENT,
        legend=dict(orientation='h', y=1.08, font=dict(size=9, color='#8899aa')),
        title=dict(text='Range Estimation — Slope & Energy Rate Drivers',
                   font=dict(size=11, color='#00d4ff', family='Rajdhani')),
    )
    fig.update_yaxes(
        title_text='Remaining Range (km)',
        range=[0, max(history['remaining_km']) * 1.15 + 1],
        gridcolor=_GRID_COLOR, color=_TICK_COLOR,
        tickfont=_TICK_FONT, secondary_y=False,
    )
    fig.update_yaxes(
        title_text='Slope (%) | Energy (Wh/km)',
        color='#ffd700',
        tickfont=dict(size=8, color='#ffd700'),
        secondary_y=True,
    )
    fig.update_xaxes(
        showgrid=False, color=_TICK_COLOR,
        tickfont=_TICK_FONT, title='Step',
    )
    return fig
