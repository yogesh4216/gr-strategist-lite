import streamlit as st
import numpy as np
import pandas as pd

# -------------------------------------------------
# PAGE SETUP (dark pit-wall style)
# -------------------------------------------------
st.set_page_config(
    page_title="GR Strategist Lite 🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #05060B;
        color: #F5F5F5;
    }
    .stMetric {
        background-color: #111319;
        border-radius: 0.5rem;
        padding: 0.3rem 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# GLOBAL CONFIG
# -------------------------------------------------
MAX_LAPS = 30
BASE_PACE = 100.0          # seconds, "fresh tire" pace
WEAR_PER_LAP = 3.5         # % tire wear per lap
PACE_PER_WEAR = 0.22       # sec lost per 1% wear
FUEL_STINT_LAPS = 20       # starting fuel in laps

# psi and temp ranges
BASE_PRESSURE_PSI = 24.0
PRESSURE_NOISE = 0.7
TEMP_BASE = 80.0      # °C
TEMP_WEAR_GAIN = 0.5  # extra °C per 1% wear
TEMP_NOISE = 3.0

# rng for synthetic simulation
RNG = np.random.default_rng(42)


# -------------------------------------------------
# STATE INIT
# -------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=[
            "lap",
            "lap_time",
            "tire_remaining",
            "fuel_laps_left",
            # tire pressures (psi)
            "fl_psi", "fr_psi", "rl_psi", "rr_psi",
            # tire surface temps (°C)
            "fl_temp", "fr_temp", "rl_temp", "rr_temp",
            # driver inputs
            "throttle_pct", "brake_pct",
            # g-forces
            "g_lat", "g_long",
            # sector stress
            "s1_stress", "s2_stress", "s3_stress",
        ]
    )
    st.session_state.current_lap = 0
    st.session_state.tire_remaining = 100.0
    st.session_state.fuel_laps_left = FUEL_STINT_LAPS


# -------------------------------------------------
# SIMULATION HELPERS
# -------------------------------------------------
def simulate_next_lap() -> dict:
    """Generate a realistic-looking next lap sample."""
    prev_lap = st.session_state.current_lap
    lap = prev_lap + 1

    # Tire wear
    wear_variation = RNG.normal(0, 0.8)
    tire_remaining = max(
        0.0, st.session_state.tire_remaining - WEAR_PER_LAP + wear_variation
    )

    # Fuel
    fuel_laps_left = max(0.0, st.session_state.fuel_laps_left - 1)

    # Driver inputs (throttle / brake)
    throttle = np.clip(RNG.normal(0.75, 0.1), 0.3, 1.0)  # 0–1
    brake = np.clip(RNG.normal(0.35, 0.1), 0.05, 1.0)

    # Pace model: slower as tire wears, slightly influenced by driver smoothness
    degradation = (100.0 - tire_remaining) * PACE_PER_WEAR
    smoothness_factor = (1.1 - throttle + 0.3 * brake)  # rough idea
    noise = RNG.normal(0, 0.5)
    lap_time = BASE_PACE + degradation * smoothness_factor + noise

    # Tire temps based on tire wear + brake usage
    base_temp = TEMP_BASE + (100.0 - tire_remaining) * TEMP_WEAR_GAIN
    fl_temp = base_temp + RNG.normal(0, TEMP_NOISE) + brake * 10
    fr_temp = base_temp + RNG.normal(0, TEMP_NOISE) + brake * 11
    rl_temp = base_temp - 5 + RNG.normal(0, TEMP_NOISE) + throttle * 4
    rr_temp = base_temp - 3 + RNG.normal(0, TEMP_NOISE) + throttle * 5

    # Tire pressures (psi). Slight drift as tire heats / wears.
    fl_psi = BASE_PRESSURE_PSI + RNG.normal(0, PRESSURE_NOISE) + (fl_temp - TEMP_BASE) * 0.02
    fr_psi = BASE_PRESSURE_PSI + RNG.normal(0, PRESSURE_NOISE) + (fr_temp - TEMP_BASE) * 0.02
    rl_psi = BASE_PRESSURE_PSI - 0.5 + RNG.normal(0, PRESSURE_NOISE) + (rl_temp - TEMP_BASE) * 0.015
    rr_psi = BASE_PRESSURE_PSI - 0.3 + RNG.normal(0, PRESSURE_NOISE) + (rr_temp - TEMP_BASE) * 0.015

    # Very simple G-force approximation
    g_lat = np.clip(RNG.normal(1.4, 0.15), 0.8, 2.5)
    g_long = np.clip(RNG.normal(1.1, 0.12), 0.6, 2.0)

    # Sector stress 0–1
    s1_stress = np.clip(0.4 + brake * 0.8 + (100 - tire_remaining) / 200, 0, 1)
    s2_stress = np.clip(0.3 + throttle * 0.5 + g_lat / 4, 0, 1)
    s3_stress = np.clip(0.5 + throttle * 0.6 + (100 - tire_remaining) / 230, 0, 1)

    return dict(
        lap=lap,
        lap_time=lap_time,
        tire_remaining=tire_remaining,
        fuel_laps_left=fuel_laps_left,
        fl_psi=fl_psi,
        fr_psi=fr_psi,
        rl_psi=rl_psi,
        rr_psi=rr_psi,
        fl_temp=fl_temp,
        fr_temp=fr_temp,
        rl_temp=rl_temp,
        rr_temp=rr_temp,
        throttle_pct=throttle * 100,
        brake_pct=brake * 100,
        g_lat=g_lat,
        g_long=g_long,
        s1_stress=s1_stress,
        s2_stress=s2_stress,
        s3_stress=s3_stress,
    )


def ensure_laps_upto(target_lap: int):
    while st.session_state.current_lap < target_lap and target_lap <= MAX_LAPS:
        row = simulate_next_lap()
        st.session_state.current_lap = int(row["lap"])
        st.session_state.tire_remaining = float(row["tire_remaining"])
        st.session_state.fuel_laps_left = float(row["fuel_laps_left"])
        st.session_state.history = pd.concat(
            [st.session_state.history, pd.DataFrame([row])],
            ignore_index=True,
        )


def predict_future(current_row: pd.Series):
    """Project future laps (for pit window + under/overcut)."""
    laps = []
    lap = int(current_row["lap"])
    tire = float(current_row["tire_remaining"])
    fuel = float(current_row["fuel_laps_left"])
    pace = float(current_row["lap_time"])

    while lap < MAX_LAPS and fuel > 0:
        lap += 1
        tire = max(0.0, tire - WEAR_PER_LAP)
        fuel = max(0.0, fuel - 1)
        degradation = (100.0 - tire) * PACE_PER_WEAR
        pace = BASE_PACE + degradation
        laps.append(
            {
                "lap": lap,
                "tire_remaining": tire,
                "fuel_laps_left": fuel,
                "pace": pace,
            }
        )

    return pd.DataFrame(laps)


def pick_pit_window(future_df: pd.DataFrame):
    """
    Pit window: tire 30–45% and fuel >= 2 laps.
    """
    if future_df.empty:
        return None, None

    window = future_df[
        (future_df["tire_remaining"] <= 45)
        & (future_df["tire_remaining"] >= 30)
        & (future_df["fuel_laps_left"] >= 2)
    ]
    if window.empty:
        # fallback: first lap where tire < 50 or fuel <= 3
        window = future_df[
            (future_df["tire_remaining"] <= 50) | (future_df["fuel_laps_left"] <= 3)
        ]
    if window.empty:
        return None, None

    # choose lap with best (lowest) pace within that window
    best_row = window.sort_values("pace").iloc[0]
    pit_lap = int(best_row["lap"])
    return max(pit_lap - 1, 1), min(pit_lap + 1, MAX_LAPS)


def under_overcut_compare(current_lap: int, pit_lap: int, future_df: pd.DataFrame):
    if pit_lap is None:
        return None

    early = max(pit_lap - 1, current_lap + 1)
    late = min(pit_lap + 1, MAX_LAPS - 1)

    def stint_cost(pit_lap_: int):
        horizon = min(pit_lap_ + 5, MAX_LAPS)
        df = future_df[future_df["lap"] <= horizon]

        before = df[df["lap"] < pit_lap_]["pace"].sum()

        # After pit: reset tire & fuel
        after_tire = 100.0
        after = 0.0
        for _ in range(pit_lap_, horizon + 1):
            degradation = (100.0 - after_tire) * PACE_PER_WEAR
            after += BASE_PACE + degradation
            after_tire = max(0.0, after_tire - WEAR_PER_LAP)
        return before + after

    early_cost = stint_cost(early)
    late_cost = stint_cost(late)
    gain = late_cost - early_cost  # positive → early better
    return dict(early_lap=early, late_lap=late, gain_seconds=gain)


def driver_consistency_score(hist: pd.DataFrame):
    if len(hist) < 3:
        return 100, "Not enough laps yet for consistency analysis."
    std = hist["lap_time"].std()
    score = max(0, 100 - std * 10)
    if score > 85:
        txt = "Super consistent – perfect for long stints ✅"
    elif score > 70:
        txt = "Good consistency – small optimizations possible 👍"
    elif score > 50:
        txt = "Aggressive / variable – might hurt tire life ⚠️"
    else:
        txt = "Highly inconsistent – risky for race stints ❌"
    return int(score), txt


def sector_heat(hist: pd.DataFrame):
    if hist.empty:
        return [0, 0, 0]
    return [
        float(hist["s1_stress"].mean()),
        float(hist["s2_stress"].mean()),
        float(hist["s3_stress"].mean()),
    ]


# -------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------
st.sidebar.title("GR Strategist Lite")
car = st.sidebar.selectbox("Car", ["GR86-035 Demo", "GR86-047 Demo"])
target_lap = st.sidebar.slider("Simulate laps up to", 5, MAX_LAPS, 15)

if st.sidebar.button("Reset stint"):
    st.session_state.history = pd.DataFrame(columns=st.session_state.history.columns)
    st.session_state.current_lap = 0
    st.session_state.tire_remaining = 100.0
    st.session_state.fuel_laps_left = FUEL_STINT_LAPS

ensure_laps_upto(target_lap)
hist = st.session_state.history.copy()

if hist.empty:
    st.info("Simulating initial laps…")
    st.stop()

current_row = hist.iloc[-1]
current_lap = int(current_row["lap"])
current_tire = float(current_row["tire_remaining"])
current_fuel = float(current_row["fuel_laps_left"])
current_lap_time = float(current_row["lap_time"])

future = predict_future(current_row)
pit_start, pit_end = pick_pit_window(future)
pit_lap_center = None
if pit_start is not None and pit_end is not None:
    pit_lap_center = (pit_start + pit_end) // 2
uo = under_overcut_compare(current_lap, pit_lap_center, future)
cons_score, cons_text = driver_consistency_score(hist)
s1, s2, s3 = sector_heat(hist)

# -------------------------------------------------
# TOP METRICS
# -------------------------------------------------
st.title("🏁 GR Strategist Lite — Real-Time Pit Strategy Assistant")
st.caption(f"{car} • Synthetic GR Cup stint • Up to {MAX_LAPS} laps")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Lap", current_lap)
m2.metric("Last Lap Time (s)", f"{current_lap_time:.3f}")
m3.metric("Tire Remaining", f"{current_tire:.1f}%")
m4.metric("Fuel Laps Left", f"{current_fuel:.1f}")

st.write(f"**Driver consistency:** {cons_score}/100 – {cons_text}")

# -------------------------------------------------
# MAIN CHARTS
# -------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    st.subheader("Lap Time Trend")
    st.line_chart(hist.set_index("lap")["lap_time"])

with c2:
    st.subheader("Tire Wear vs Laps")
    st.line_chart(hist.set_index("lap")["tire_remaining"])

# -------------------------------------------------
# TIRE & TELEMETRY MODULE
# -------------------------------------------------
st.markdown("---")
st.subheader("🛞 Tire & Telemetry Snapshot")

# Tire corner metrics
t1, t2, t3, t4 = st.columns(4)
t1.metric("FL Pressure (psi)", f"{current_row['fl_psi']:.1f}")
t2.metric("FR Pressure (psi)", f"{current_row['fr_psi']:.1f}")
t3.metric("RL Pressure (psi)", f"{current_row['rl_psi']:.1f}")
t4.metric("RR Pressure (psi)", f"{current_row['rr_psi']:.1f}")

u1, u2, u3, u4 = st.columns(4)
u1.metric("FL Temp (°C)", f"{current_row['fl_temp']:.1f}")
u2.metric("FR Temp (°C)", f"{current_row['fr_temp']:.1f}")
u3.metric("RL Temp (°C)", f"{current_row['rl_temp']:.1f}")
u4.metric("RR Temp (°C)", f"{current_row['rr_temp']:.1f}")

# Throttle / Brake / G-forces
tb1, tb2, tb3 = st.columns(3)
tb1.metric("Throttle Avg (%)", f"{current_row['throttle_pct']:.0f}")
tb2.metric("Brake Avg (%)", f"{current_row['brake_pct']:.0f}")
tb3.metric("G-Forces (Lat / Long)", f"{current_row['g_lat']:.1f} / {current_row['g_long']:.1f}")

# -------------------------------------------------
# STRATEGY ENGINE
# -------------------------------------------------
st.markdown("---")
st.subheader("🧠 Strategy Engine — Pit Window & Under/Overcut")

if pit_start is None:
    st.info("Not enough forecast to recommend a pit window yet.")
else:
    st.write(
        f"🔍 **Recommended pit window:** laps **{pit_start}–{pit_end}** "
        f"(balancing tire life, pace and remaining fuel)."
    )
    if uo:
        gain = uo["gain_seconds"]
        early = uo["early_lap"]
        late = uo["late_lap"]
        if gain > 0:
            st.success(
                f"📈 Undercut advantage: pitting on **lap {early}** is estimated "
                f"~**{gain:.1f}s faster** over the next stint than staying out until lap {late}."
            )
        elif gain < 0:
            st.warning(
                f"📉 Overcut advantage: staying out until **lap {late}** is estimated "
                f"~**{abs(gain):.1f}s faster** than pitting early on lap {early}."
            )
        else:
            st.info("⚖️ Under vs overcut are roughly equivalent for the next stint horizon.")

# -------------------------------------------------
# ALERTS
# -------------------------------------------------
st.markdown("---")
st.subheader("⚠ Live Alerts")

low_pressure = (
    current_row["fl_psi"] < 21
    or current_row["fr_psi"] < 21
    or current_row["rl_psi"] < 21
    or current_row["rr_psi"] < 21
)
overheated = (
    current_row["fl_temp"] > 115
    or current_row["fr_temp"] > 115
    or current_row["rl_temp"] > 115
    or current_row["rr_temp"] > 115
)
low_fuel = current_fuel <= 3

if low_pressure:
    st.error("🟥 Tire pressure critical – possible slow puncture. **PIT NOW**.")
if overheated:
    st.warning("🟧 Tire temperatures very high – back off or PIT soon.")
if low_fuel:
    st.error("⛽ Fuel critically low – **pit within next 1–2 laps**.")

if not (low_pressure or overheated or low_fuel):
    st.success("✅ No critical issues – car is healthy. Free to extend stint.")

# -------------------------------------------------
# TRACK HEAT MAP
# -------------------------------------------------
st.markdown("---")
st.subheader("🔥 Track Sector Stress Map")

heat_df = pd.DataFrame(
    {
        "Sector": ["S1 – Heavy Braking", "S2 – High Speed", "S3 – Traction Zones"],
        "Stress": [s1, s2, s3],
    }
)
st.bar_chart(heat_df.set_index("Sector"))

for sector, stress in zip(heat_df["Sector"], heat_df["Stress"]):
    if stress > 0.8:
        st.write(f"🟥 {sector}: very high stress – protect tires here.")
    elif stress > 0.6:
        st.write(f"🟧 {sector}: moderate–high stress – manage inputs.")
    else:
        st.write(f"🟩 {sector}: low–medium stress – safe for pushing.")

# -------------------------------------------------
# RAW DATA TABLE
# -------------------------------------------------
with st.expander("Show underlying lap data"):
    st.dataframe(hist.reset_index(drop=True))
