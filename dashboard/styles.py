"""
styles.py — All CSS for E-Scooter Digital Twin Dashboard
"""

DASHBOARD_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Rajdhani:wght@400;500;600;700&display=swap');

    :root {
        --bg-primary:    #0a0e1a;
        --bg-secondary:  #111827;
        --bg-card:       #1a2235;
        --accent-cyan:   #00d4ff;
        --accent-green:  #00ff88;
        --accent-orange: #ff6b35;
        --accent-red:    #ff3366;
        --accent-yellow: #ffd700;
        --text-primary:  #e8f0fe;
        --text-secondary:#8899aa;
        --border:        #2a3a4a;
    }

    /* ── App base ── */
    .stApp { background: var(--bg-primary); font-family: 'Rajdhani', sans-serif; }
    .main .block-container { padding: 1rem 2rem; max-width: 100%; }

    /* ── Header ── */
    .dt-header {
        background: linear-gradient(135deg, #0a0e1a 0%, #1a2235 50%, #0a0e1a 100%);
        border: 1px solid var(--accent-cyan);
        border-radius: 12px;
        padding: 20px 30px;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
    }
    .dt-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent-cyan), var(--accent-green), transparent);
    }
    .dt-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2rem; font-weight: 700;
        color: var(--accent-cyan);
        letter-spacing: 3px; text-transform: uppercase; margin: 0;
    }
    .dt-subtitle {
        color: var(--text-secondary);
        font-size: 0.9rem; letter-spacing: 2px; margin: 4px 0 0 0;
    }
    .dt-status {
        display: inline-block;
        background: rgba(0,255,136,0.1);
        border: 1px solid var(--accent-green);
        color: var(--accent-green);
        padding: 4px 12px; border-radius: 20px;
        font-size: 0.8rem; letter-spacing: 1px;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 16px 20px;
        position: relative; overflow: hidden;
    }
    .metric-card::before {
        content: ''; position: absolute;
        top: 0; left: 0;
        width: 3px; height: 100%;
        background: var(--accent-cyan);
    }
    .metric-card.warning::before { background: var(--accent-orange); }
    .metric-card.danger::before  { background: var(--accent-red); }
    .metric-card.good::before    { background: var(--accent-green); }

    .metric-label {
        font-size: 0.7rem; letter-spacing: 2px;
        color: var(--text-secondary);
        text-transform: uppercase; margin-bottom: 6px;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem; font-weight: 700;
        color: var(--text-primary); line-height: 1;
    }
    .metric-unit  { font-size: 0.8rem; color: var(--text-secondary); margin-left: 4px; }
    .metric-sub   { font-size: 0.75rem; color: var(--text-secondary); margin-top: 4px; }

    /* ── Section headers ── */
    .section-header {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.75rem; letter-spacing: 3px;
        text-transform: uppercase;
        color: var(--accent-cyan);
        border-bottom: 1px solid var(--border);
        padding-bottom: 6px; margin: 16px 0 12px 0;
    }

    /* ── Alert banners ── */
    .alert-banner {
        background: rgba(255,51,102,0.1);
        border: 1px solid var(--accent-red);
        border-radius: 8px; padding: 10px 16px;
        color: var(--accent-red);
        font-size: 0.85rem; letter-spacing: 1px; margin: 4px 0;
    }
    .warn-banner {
        background: rgba(255,107,53,0.1);
        border: 1px solid var(--accent-orange);
        border-radius: 8px; padding: 10px 16px;
        color: var(--accent-orange);
        font-size: 0.85rem; letter-spacing: 1px; margin: 4px 0;
    }
    .ok-banner {
        background: rgba(0,255,136,0.05);
        border: 1px solid var(--accent-green);
        border-radius: 8px; padding: 10px 16px;
        color: var(--accent-green);
        font-size: 0.85rem; letter-spacing: 1px; margin: 4px 0;
    }

    /* ── Step badge ── */
    .step-badge {
        background: var(--bg-card);
        border: 1px solid var(--accent-cyan);
        border-radius: 8px; padding: 8px 16px;
        font-family: 'JetBrains Mono', monospace;
        color: var(--accent-cyan); font-size: 0.9rem;
        display: inline-block;
    }

    /* ── Flow diagram boxes ── */
    .flow-box {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px; padding: 10px;
        text-align: center;
        font-size: 0.75rem; letter-spacing: 1px;
        color: var(--text-secondary);
    }
    .flow-box.active {
        border-color: var(--accent-cyan);
        color: var(--accent-cyan);
    }

    /* ── Live input display boxes ── */
    .live-input-box {
        background: var(--bg-card);
        padding: 12px 16px;
        border-radius: 8px;
        border: 1px solid var(--border);
        margin-bottom: 8px;
    }
    .live-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem; font-weight: 700;
    }

    /* ── Summary card (CSV player) ── */
    .summary-card {
        background: var(--bg-card);
        border: 1px solid var(--accent-cyan);
        border-radius: 12px;
        padding: 20px 24px;
        margin: 8px 0;
    }
    .summary-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.8rem; letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--accent-cyan);
        margin-bottom: 12px;
        border-bottom: 1px solid var(--border);
        padding-bottom: 6px;
    }
    .summary-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem; font-weight: 700;
        color: var(--text-primary);
    }
    .summary-label {
        font-size: 0.7rem; letter-spacing: 1px;
        color: var(--text-secondary);
        text-transform: uppercase;
        margin-top: 4px;
    }

    /* ── Hide Streamlit UI chrome ── */
    #MainMenu {visibility: hidden;}
    footer     {visibility: hidden;}
    /* NOTE: header is NOT hidden — it contains the sidebar toggle button */
</style>
"""


def inject_css():
    """Call this once at the top of any Streamlit page."""
    import streamlit as st
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)
