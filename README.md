# E-Scooter Digital Twin (Physics + Data)

Data-driven digital twin for an e-scooter. The core implementation is in the All_Files folder and combines physics-derived features with real driving data to run a four-model feedback loop (controller -> motor -> dynamics -> range).

## Physics + data usage (what the code actually does)

### Vehicle driving data (BMW i3, 10 Hz)
Source: BMW i3 TripA/TripB cycles. Preprocessing in [All_Files/preprocess_bmw_i3.py](All_Files/preprocess_bmw_i3.py).

Physics-derived signals:
- Slope from per-step elevation change (bounded to realistic grades).
- Distance and energy per step from velocity, voltage, current, and $\Delta t$.
- Mass-normalized energy (Wh/kg) to transfer BMW-scale energy use to scooter-scale inference.
- Trip-local lag features (no cross-trip leakage).

Core equations used:
$$
v_{m/s} = \frac{v_{km/h}}{3.6}
$$
$$
\Delta s = v_{m/s} \cdot \Delta t
$$
$$
	ext{slope} = \frac{\Delta h}{\Delta s}\quad\text{(clipped to }[-0.30, 0.30]\text{)}
$$
$$
P_{bat} = -V \cdot I
$$
$$
E_{step} = \frac{P_{bat} \cdot \Delta t}{3600},\quad E_{norm} = \frac{E_{step}}{m_{BMW}}
$$

Outputs in Datasets/processed: vehicle_dynamics_* and range_management_* splits.

### Motor + controller data (Paderborn PMSM, 2 Hz)
Source: Paderborn PMSM measurements. Preprocessing in [All_Files/preprocess_paderborn.py](All_Files/preprocess_paderborn.py).

Physics-derived signals:
- Throttle proxy from $i_q$ (FOC torque current).
- Duty cycle from $|V_{phase}| / V_{bus}$.
- Load torque from motor dynamics.
- Electrical and mechanical power; efficiency proxy.
- Session-local lag features (no cross-session leakage).

Core equations used:
$$
	ext{throttle\_proxy} = 100 \cdot \frac{i_q}{\max(|i_q|)}
$$
$$
|V_{phase}| = \sqrt{u_d^2 + u_q^2},\quad D = \frac{|V_{phase}|}{V_{bus}}
$$
$$
\omega = \frac{2\pi}{60} \cdot \text{RPM}
$$
$$
T_{load} = T_{em} - J\frac{d\omega}{dt} - B\omega
$$
$$
P_{elec} = u_d i_d + u_q i_q,\quad P_{mech} = T_{em} \cdot \omega
$$

Outputs in Datasets/processed: motor_* splits.

## Model stack used by the digital twin (All_Files)
The current loop used by the dashboard is defined in [All_Files/dashboard_code/models.py](All_Files/dashboard_code/models.py).

1) Controller (XGBoost) — [All_Files/model_controller.py](All_Files/model_controller.py)
2) Motor LSTM (seq=50) — [All_Files/model_motor.py](All_Files/model_motor.py)
3) Dynamics LSTM (seq=150) — [All_Files/model_dynamics_lstm.py](All_Files/model_dynamics_lstm.py)
4) Range LSTM (seq=200) — [All_Files/model_range_lstm.py](All_Files/model_range_lstm.py)

Key physics constraints during inference:
- Dynamics LSTM runs in BMW-scale velocity, then scales to scooter display speed.
$$
v_{scooter} = v_{BMW} \cdot \frac{25}{150}
$$
- SoC monotonicity is enforced post-prediction (regen allows small increases).

## Range label physics (remaining_km)
Range LSTM uses a derived label based on trip-level efficiency (stable vs rolling efficiency noise). See [All_Files/model_range_lstm.py](All_Files/model_range_lstm.py).

$$
E_{tot} = \sum |E_{norm}| \cdot m_{BMW} \cdot \Delta t
$$
$$
d_{tot} = \sum \frac{v}{3.6} \cdot \Delta t
$$
$$
\eta_{trip} = \frac{E_{tot}}{d_{tot}},\quad \eta_{scooter} = \eta_{trip} \cdot \frac{m_{scooter}}{m_{BMW}}
$$
$$
	ext{remaining\_km}(t) = \frac{\text{SoC}(t)}{100} \cdot \frac{E_{bat}}{\eta_{scooter}}
$$

## Scenario inputs for the dashboard
The dashboard expects a CSV with:
step, time_s, throttle, slope, regen, ambient_temp, trip_type

Scenario generators live in Ride_data_generation (outside All_Files).

## Validation and drift checks
- Closed-loop validation: [All_Files/validate_digital_twin.py](All_Files/validate_digital_twin.py)
- Drift confirmation per trip: [All_Files/confirm_drift.py](All_Files/confirm_drift.py)
- Quick controller check: [All_Files/valid_MP.py](All_Files/valid_MP.py)
- Ad-hoc data checks: [All_Files/correction_vehicle_dynamics.py](All_Files/correction_vehicle_dynamics.py)

## Dashboard (All_Files)
Streamlit app and UI modules:
- [All_Files/dashboard_code/app.py](All_Files/dashboard_code/app.py)
- [All_Files/dashboard_code/models.py](All_Files/dashboard_code/models.py)
- [All_Files/dashboard_code/charts.py](All_Files/dashboard_code/charts.py)
- [All_Files/dashboard_code/state.py](All_Files/dashboard_code/state.py)
- [All_Files/dashboard_code/styles.py](All_Files/dashboard_code/styles.py)

Run the dashboard:
```bash
streamlit run app.py
```
Run from the All_Files/dashboard_code folder.

## Outputs (artifacts)
Training scripts write models and plots to the top-level Models/ and Plots/ folders.
- Models: scalers, configs, and trained models (joblib/keras)
- Plots: training curves, predictions vs actual, and validation figures

## EDA
Exploratory analysis for both datasets in [All_Files/eda_escooter_dt.py](All_Files/eda_escooter_dt.py).
