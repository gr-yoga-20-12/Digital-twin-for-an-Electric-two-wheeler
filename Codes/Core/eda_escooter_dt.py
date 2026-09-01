import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import glob

# ─────────────────────────────────────────────
#  CONFIGURE FILE NAMES HERE
# ─────────────────────────────────────────────
PADERBORN_FILE = "measures_v2.csv"

# DualEMobility may have multiple CSVs inside a folder
# Set this to the escooter CSV filename or folder name
DUAL_FILE = "escooter"   # can be a .csv or a folder name

# ─────────────────────────────────────────────

SEP = "=" * 60

def section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def load_dual(path):
    """Load DualEMobility — handles single CSV or folder of CSVs."""
    if os.path.isdir(path):
        csvs = glob.glob(os.path.join(path, "**/*.csv"), recursive=True)
        if not csvs:
            raise FileNotFoundError(f"No CSVs found inside folder: {path}")
        frames = []
        for f in csvs:
            tmp = pd.read_csv(f)
            tmp["_source_file"] = os.path.basename(f)
            frames.append(tmp)
        df = pd.concat(frames, ignore_index=True)
        print(f"  Loaded {len(csvs)} CSV(s) from folder '{path}'")
        print(f"  Files: {[os.path.basename(f) for f in csvs]}")
        return df
    else:
        # Try exact name first, then with .csv extension
        if os.path.exists(path):
            return pd.read_csv(path)
        elif os.path.exists(path + ".csv"):
            return pd.read_csv(path + ".csv")
        else:
            raise FileNotFoundError(f"Cannot find: {path} or {path}.csv")


# ══════════════════════════════════════════════
#  DATASET 1 — PADERBORN PMSM
# ══════════════════════════════════════════════

section("DATASET 1 — PADERBORN PMSM")

try:
    df_p = pd.read_csv(PADERBORN_FILE)
    print(f"\n  Rows: {len(df_p):,}   Columns: {len(df_p.columns)}")

    # Q1 — Column names and types
    section("Q1 — Column names & data types")
    print(df_p.dtypes.to_string())

    # Q2 — Missing values
    section("Q2 — Missing values")
    nulls = df_p.isnull().sum()
    null_pct = (nulls / len(df_p) * 100).round(2)
    null_report = pd.DataFrame({"null_count": nulls, "null_%": null_pct})
    print(null_report[null_report["null_count"] > 0].to_string()
          if null_report["null_count"].sum() > 0
          else "  No missing values found.")

    # Q3 — Sampling rate
    section("Q3 — Sampling rate")
    if "profile_id" in df_p.columns:
        # Compute dt within first session only
        first_pid = df_p["profile_id"].iloc[0]
        session0  = df_p[df_p["profile_id"] == first_pid].copy()
        if len(session0) > 1:
            # Use row index as proxy for time if no time column
            if "time" in df_p.columns:
                dt_vals = session0["time"].diff().dropna()
            else:
                dt_vals = pd.Series([1.0])   # fallback
            dt_median = dt_vals.median()
            print(f"  Median Δt between rows : {dt_median:.4f}  units")
            print(f"  Estimated sampling rate : {1/dt_median:.2f} Hz  (rows/unit)")
            print(f"  NOTE — confirm 'time' column units (seconds vs ms)")
        else:
            print("  Only 1 row in first session — cannot estimate dt")
    else:
        print("  No 'profile_id' column found — estimating from full dataset")
        if "time" in df_p.columns:
            dt_median = df_p["time"].diff().dropna().median()
            print(f"  Median Δt : {dt_median:.4f}")
        else:
            print("  No 'time' column found either. Row index = time proxy.")

    # Q4 — Profile sessions
    section("Q4 — Profile sessions (profile_id)")
    if "profile_id" in df_p.columns:
        sess = df_p.groupby("profile_id").size().reset_index(name="row_count")
        sess["approx_duration"] = sess["row_count"].apply(
            lambda x: f"{x} rows"
        )
        print(f"  Total sessions : {len(sess)}")
        print(sess.to_string(index=False))
    else:
        print("  'profile_id' column not found.")

    # Q5 — Value ranges
    section("Q5 — Value ranges (key columns)")
    key_cols_p = [c for c in
                  ["u_d","u_q","i_d","i_q","motor_speed","torque",
                   "pm","stator_winding","stator_tooth","stator_yoke",
                   "ambient","coolant"]
                  if c in df_p.columns]
    print(df_p[key_cols_p].describe().round(3).to_string())

    # ── Plots ──
    section("PLOTS — Paderborn distributions")
    n = len(key_cols_p)
    cols_per_row = 4
    rows_needed  = (n + cols_per_row - 1) // cols_per_row
    fig, axes = plt.subplots(rows_needed, cols_per_row,
                             figsize=(16, 4 * rows_needed))
    axes = axes.flatten()
    for i, col in enumerate(key_cols_p):
        axes[i].hist(df_p[col].dropna(), bins=60, color="#7F77DD", edgecolor="none")
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel("value")
        axes[i].set_ylabel("count")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Paderborn PMSM — column distributions", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("eda_paderborn_distributions.png", dpi=120, bbox_inches="tight")
    print("  Saved: eda_paderborn_distributions.png")

    # Motor speed over time (first session)
    if "profile_id" in df_p.columns and "motor_speed" in df_p.columns:
        fig2, ax2 = plt.subplots(figsize=(14, 4))
        for pid, grp in df_p.groupby("profile_id"):
            ax2.plot(grp.index, grp["motor_speed"], linewidth=0.6, label=f"pid {pid}")
        ax2.set_title("Motor speed across all sessions")
        ax2.set_xlabel("row index")
        ax2.set_ylabel("motor_speed")
        ax2.legend(fontsize=7, ncol=5)
        plt.tight_layout()
        plt.savefig("eda_paderborn_speed_sessions.png", dpi=120, bbox_inches="tight")
        print("  Saved: eda_paderborn_speed_sessions.png")

    print("\n  ✅ Paderborn EDA complete.")

except FileNotFoundError as e:
    print(f"\n  ❌ File not found: {e}")
    print(f"     Make sure '{PADERBORN_FILE}' is in the same folder as this script.")


# ══════════════════════════════════════════════
#  DATASET 2 — DualEMobility E-Scooter
# ══════════════════════════════════════════════

section("DATASET 2 — DualEMobility E-SCOOTER")

try:
    df_d = load_dual(DUAL_FILE)
    print(f"\n  Rows: {len(df_d):,}   Columns: {len(df_d.columns)}")

    # Q1 — Columns
    section("Q1 — Column names & data types")
    print(df_d.dtypes.to_string())

    # Q2 — Missing values
    section("Q2 — Missing values")
    nulls_d   = df_d.isnull().sum()
    null_pct_d = (nulls_d / len(df_d) * 100).round(2)
    null_rep_d = pd.DataFrame({"null_count": nulls_d, "null_%": null_pct_d})
    print(null_rep_d[null_rep_d["null_count"] > 0].to_string()
          if null_rep_d["null_count"].sum() > 0
          else "  No missing values found.")

    # Q3 — Sampling rate
    section("Q3 — Sampling rate")
    time_candidates = [c for c in df_d.columns
                       if any(k in c.lower() for k in ["time","timestamp","ts","date"])]
    if time_candidates:
        tcol = time_candidates[0]
        print(f"  Using column '{tcol}' as time reference")
        try:
            df_d[tcol] = pd.to_datetime(df_d[tcol])
            dt_d = df_d[tcol].diff().dt.total_seconds().dropna()
            print(f"  Median Δt : {dt_d.median():.2f} seconds")
            print(f"  Min Δt    : {dt_d.min():.2f} s")
            print(f"  Max Δt    : {dt_d.max():.2f} s")
        except Exception:
            print(f"  Could not parse '{tcol}' as datetime.")
            dt_d = df_d[tcol].diff().dropna()
            print(f"  Median Δt (raw) : {dt_d.median():.4f}")
    else:
        print("  No timestamp column found. Columns available:")
        print(f"  {list(df_d.columns)}")

    # Q4 — SoC range
    section("Q4 — SoC range")
    soc_candidates = [c for c in df_d.columns if "soc" in c.lower()]
    if soc_candidates:
        for sc in soc_candidates:
            print(f"  '{sc}' → min: {df_d[sc].min():.2f}  "
                  f"max: {df_d[sc].max():.2f}  "
                  f"mean: {df_d[sc].mean():.2f}")
        print(f"\n  Scale check: {'0–1 scale' if df_d[soc_candidates[0]].max() <= 1.1 else '0–100 scale'}")
    else:
        print("  No SoC column found. Columns:")
        print(f"  {list(df_d.columns)}")

    # Q5 — Value ranges
    section("Q5 — Value ranges (all numeric columns)")
    num_cols_d = df_d.select_dtypes(include=[np.number]).columns.tolist()
    print(df_d[num_cols_d].describe().round(3).to_string())

    # GPS coverage check
    section("GPS coverage")
    lat_cols = [c for c in df_d.columns if "lat" in c.lower()]
    lon_cols = [c for c in df_d.columns if "lon" in c.lower() or "lng" in c.lower()]
    if lat_cols and lon_cols:
        lat, lon = lat_cols[0], lon_cols[0]
        print(f"  Lat range : {df_d[lat].min():.5f} → {df_d[lat].max():.5f}")
        print(f"  Lon range : {df_d[lon].min():.5f} → {df_d[lon].max():.5f}")
        print(f"  GPS null% : lat={df_d[lat].isnull().mean()*100:.1f}%  "
              f"lon={df_d[lon].isnull().mean()*100:.1f}%")
    else:
        print("  No lat/lon columns found.")

    # ── Plots ──
    section("PLOTS — DualEMobility distributions")
    plot_cols_d = num_cols_d[:12]   # max 12 to keep plot readable
    n2 = len(plot_cols_d)
    rows2 = (n2 + 3) // 4
    fig3, axes3 = plt.subplots(rows2, 4, figsize=(16, 4 * rows2))
    axes3 = axes3.flatten()
    for i, col in enumerate(plot_cols_d):
        axes3[i].hist(df_d[col].dropna(), bins=50, color="#1D9E75", edgecolor="none")
        axes3[i].set_title(col, fontsize=10)
        axes3[i].set_xlabel("value")
        axes3[i].set_ylabel("count")
    for j in range(i + 1, len(axes3)):
        axes3[j].set_visible(False)
    plt.suptitle("DualEMobility E-Scooter — column distributions", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("eda_dual_distributions.png", dpi=120, bbox_inches="tight")
    print("  Saved: eda_dual_distributions.png")

    # SoC over time
    if soc_candidates and time_candidates:
        fig4, ax4 = plt.subplots(figsize=(14, 4))
        ax4.plot(df_d[time_candidates[0]], df_d[soc_candidates[0]],
                 color="#1D9E75", linewidth=0.8)
        ax4.set_title("SoC over time — DualEMobility E-Scooter")
        ax4.set_xlabel("timestamp")
        ax4.set_ylabel(soc_candidates[0])
        plt.tight_layout()
        plt.savefig("eda_dual_soc_timeseries.png", dpi=120, bbox_inches="tight")
        print("  Saved: eda_dual_soc_timeseries.png")

    print("\n  ✅ DualEMobility EDA complete.")

except FileNotFoundError as e:
    print(f"\n  ❌ File not found: {e}")
    print(f"     Set DUAL_FILE at the top of the script to your exact filename or folder name.")


section("EDA COMPLETE — next steps")
print("""
  1. Check printed column names match what we planned
  2. Note the sampling rate (Δt) for both datasets
  3. Check SoC scale (0-1 or 0-100)
  4. Count profile_id sessions in Paderborn
  5. Paste the printed output back — we'll fix anything unexpected
     before writing the preprocessing script.
""")
