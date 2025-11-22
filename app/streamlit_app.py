import streamlit as st
import numpy as np
import pandas as pd

# -------------------------------------------------
# PAGE SETUP
# -------------------------------------------------
st.set_page_config(
    page_title="GR Strategist Lite 🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Simple dark styling
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
BASE_PACE = 100.0          # sec, baseline lap time
WEAR_PER_LAP = 3.5         # % tire wear per lap
PACE_PER_WEAR = 0.22       # sec of pace loss per 1% wear

FUEL_STINT_LAPS = 20       # how many laps a full tank can do
FUEL_PERCENT_PER_LAP = 100.0 / FUEL_STINT_LAPS

BASE_PRESSURE_PSI = 24.0
PRESSURE_NOISE = 0.7

TEMP_BASE = 80.0           # °C
TEMP_WEAR_GAIN = 0.5       # extra °C per 1% wear
TEMP_NOISE = 3.0

RNG = np.random.default_rng(42)

# -------------------------------------------------
# RESET OLD INCOMPATIBLE STATE
# -------------------------------------------------
if "history" in st.session_state:
    cols = list(st.session_state["history"].columns)
    if "fuel_percent" not in cols:
        st.session_state.clear()

# -------------------------------------------------
# INIT STATE
# -------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(
        columns=[
            "lap",
            "lap_time",
            "tire_remaining",
            "fuel_percent",
            "fl_psi",
            "fr_psi",
            "rl_psi",
            "rr_psi",
            "fl_temp",
            "fr_temp",
            "rl_temp",
            "rr_temp",
            "throttle_pct",
            "brake_pct",
            "g_lat",
            "g_long",
            "s1_stress",
            "s2_stress",
            "s3_stress",
        ]
    )
    st.session_state.current_lap = 0
    st.session_state.tire_remaining = 100.0
    st.session_state.fuel_percent = 100.0


# -------------------------------------------------
# SIMULATION HELPERS
# -------------------------------------------------
def simulate_next_lap() -> dict:
    """Generate the next lap with realistic-looking telemetry."""
    prev_lap = st.session_state.current_lap
    lap = prev_lap + 1

    # Tire wear
    wear_variation = RNG.normal(0, 0.8)
    tire_remaining = max(
        0.0, st.session_state.tire_remaining - WEAR_PER_LAP + wear_variation
    )

    # Fuel % usage
    fuel_percent = max(
        0.0, st.session_state.fuel_percent - FUEL_PERCENT_PER_LAP
    )

    # Driver inputs
    throttle = np.clip(RNG.normal(0.75, 0.1), 0.3, 1.0)  # 0–1
    brake = np.clip(RNG.normal(0.35, 0.1), 0.05, 1.0)

    # Pace model
    degradation = (100.0 - tire_remaining) * PACE_PER_WEAR
    smoothness_factor = (1.1 - throttle + 0.3 * brake)
    noise = RNG.normal(0, 0.5)
    lap_time = BASE_PACE + degradation * smoothness_factor + noise

    # Tire temps
    base_temp = TEMP_BASE + (100.0 - tire_remaining) * TEMP_WEAR_GAIN
    fl_temp = base_temp + RNG.normal(0, TEMP_NOISE) + brake * 10
    fr_temp = base_temp + RNG.normal(0, TEMP_NOISE) + brake * 11
    rl_temp = base_temp - 5 + RNG.normal(0, TEMP_NOISE) + throttle * 4
    rr_temp = base_temp - 3 + RNG.normal(0, TEMP_NOISE) + throttle * 5

    # Tire pressures (psi)
    fl_psi = BASE_PRESSURE_PSI + RNG.normal(0, PRESSURE_NOISE) + (fl_temp - TEMP_BASE) * 0.02
    fr_psi = BASE_PRESSURE_PSI + RNG.normal(0, PRESSURE_NOISE) + (fr_temp - TEMP_BASE) * 0.02
    rl_psi = BASE_PRESSURE_PSI - 0.5 + RNG.normal(0, PRESSURE_NOISE) + (rl_temp - TEMP_BASE) * 0.015
    rr_psi = BASE_PRESSURE_PSI - 0.3 + RNG.normal(0, PRESSURE_NOISE) + (rr_temp - TEMP_BASE) * 0.015

    # G-forces
    g_lat = np.clip(RNG.normal(1.4, 0.15), 0.8, 2.5)
    g_long = np.clip(RNG.normal(1.1, 0.12), 0.6, 2.0)

    # Sector stress
    s1_stress = np.clip(0.4 + brake * 0.8 + (100 - tire_remaining) / 200, 0, 1)
    s2_stress = np.clip(0.3 + throttle * 0.5 + g_lat / 4, 0, 1)
    s3_stress = np.clip(0.5 + throttle * 0.6 + (100 - tire_remaining) / 230, 0, 1)

    return dict(
        lap=lap,
        lap_time=lap_time,
        tire_remaining=tire_remaining,
        fuel_percent=fuel_percent,
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
        st.session_state.fuel_percent = float(row["fuel_percent"])
        st.session_state.history = pd.concat(
            [st.session_state.history, pd.DataFrame([row])],
            ignore_index=True,
        )


def predict_future(current_row: pd.Series):
    """Project future laps from current telemetry."""
    laps = []
    lap = int(current_row["lap"])
    tire = float(current_row["tire_remaining"])
    fuel_pct = float(current_row["fuel_percent"])
    pace = float(current_row["lap_time"])

    while lap < MAX_LAPS and fuel_pct > 0:
        lap += 1
        tire = max(0.0, tire - WEAR_PER_LAP)
        fuel_pct = max(0.0, fuel_pct - FUEL_PERCENT_PER_LAP)
        degradation = (100.0 - tire) * PACE_PER_WEAR
        pace = BASE_PACE + degradation
        fuel_laps_left = fuel_pct / FUEL_PERCENT_PER_LAP if FUEL_PERCENT_PER_LAP > 0 else 0.0
        laps.append(
            {
                "lap": lap,
                "tire_remaining": tire,
                "fuel_percent": fuel_pct,
                "fuel_laps_left": fuel_laps_left,
                "pace": pace,
            }
        )

    return pd.DataFrame(laps)


def pick_pit_window(future_df: pd.DataFrame):
    """Pick pit window using tire life + fuel."""
    if future_df.empty:
        return None, None

    window = future_df[
        (future_df["tire_remaining"] <= 45)
        & (future_df["tire_remaining"] >= 30)
        & (future_df["fuel_laps_left"] >= 2)
    ]

    if window.empty:
        window = future_df[
            (future_df["tire_remaining"] <= 50) | (future_df["fuel_laps_left"] <= 3)
        ]

    if window.empty:
        return None, None

    best_row = window.sort_values("pace").iloc[0]
    pit_lap = int(best_row["lap"])
    return max(pit_lap - 1, 1), min(pit_lap + 1, MAX_LAPS)


def under_overcut_compare(current_lap: int, pit_lap: int, future_df: pd.DataFrame):
    if pit_lap is None:
        return None

    early = max(pit_lap - 1, current_lap + 1)
    late = min(pit_lap + 1, MAX_LAPS - 1)

    def stint_cost(pit_lap_):
        horizon = min(pit_lap_ + 5, MAX_LAPS)
        df = future_df[future_df["lap"] <= horizon]

        before = df[df["lap"] < pit_lap_]["pace"].sum()

        after_tire = 100.0
        after = 0.0
        for _ in range(pit_lap_, horizon + 1):
            degradation = (100.0 - after_tire) * PACE_PER_WEAR
            after += BASE_PACE + degradation
            after_tire = max(0.0, after_tire - WEAR_PER_LAP)
        return before + after

    early_cost = stint_cost(early)
    late_cost = stint_cost(late)
    gain = late_cost - early_cost
    return dict(early_lap=early, late_lap=late, gain_seconds=gain)


def driver_consistency_score(hist: pd.DataFrame):
    if len(hist) < 3:
        return 100, "Not enough laps yet for consistency analysis."
    std = hist["lap_time"].std()
    score = max(0, 100 - std * 10)
    if score > 85:
        txt = "Super consistent – ideal for long stints ✅"
    elif score > 70:
        txt = "Good consistency – small optimizations possible 👍"
    elif score > 50:
        txt = "Aggressive / variable – may hurt tire life ⚠️"
    else:
        txt = "Highly inconsistent – risky for race pace ❌"
    return int(score), txt


def sector_heat(hist: pd.DataFrame):
    if hist.empty:
        return [0, 0, 0]
    return [
        float(hist["s1_stress"].mean()),
        float(hist["s2_stress"].mean()),
        float(hist["s3_stress"].mean()),
    ]


def aggression_level(row: pd.Series) -> float:
    """0–100 attack score based on throttle, brake, g-forces."""
    base = 0.5 * (row["throttle_pct"] / 100) + 0.3 * (row["brake_pct"] / 100)
    base += 0.2 * (row["g_lat"] / 2.0)
    return float(np.clip(base * 100, 0, 100))


def colored_value(val, low_ok=True, unit=""):
    """Return a text with emoji color style based on thresholds."""
    if low_ok:
        if val < 30:
            icon = "🟢"
        elif val < 60:
            icon = "🟡"
        else:
            icon = "🟥"
    else:
        if val < 40:
            icon = "🟥"
        elif val < 60:
            icon = "🟡"
        else:
            icon = "🟢"
    return f"{icon} {val:.1f}{unit}"


# -------------------------------------------------
# SIDEBAR NAV
# -------------------------------------------------
st.sidebar.title("GR Strategist Lite")
car = st.sidebar.selectbox("Car", ["GR86-035 Demo", "GR86-047 Demo"])
target_lap = st.sidebar.slider("Simulate laps up to", 5, MAX_LAPS, 15)

page = st.sidebar.radio(
    "View",
    ["Race HUD", "Strategy & Pit Window", "Telemetry & Driver", "Track & Weather", "Pit Loss Tool"],
)

if st.sidebar.button("Reset stint"):
    st.session_state.clear()
    st.experimental_rerun()

ensure_laps_upto(target_lap)
hist = st.session_state.history.copy()

if hist.empty:
    st.info("Simulating initial laps…")
    st.stop()

current_row = hist.iloc[-1]
current_lap = int(current_row["lap"])
current_tire = float(current_row["tire_remaining"])
fuel_percent = float(current_row["fuel_percent"])
fuel_laps_left = fuel_percent / FUEL_PERCENT_PER_LAP if FUEL_PERCENT_PER_LAP > 0 else 0.0
current_lap_time = float(current_row["lap_time"])
attack = aggression_level(current_row)

future = predict_future(current_row)
pit_start, pit_end = pick_pit_window(future)
pit_center = None
if pit_start is not None and pit_end is not None:
    pit_center = (pit_start + pit_end) // 2
uo = under_overcut_compare(current_lap, pit_center, future)
cons_score, cons_text = driver_consistency_score(hist)
s1, s2, s3 = sector_heat(hist)

# -------------------------------------------------
# PAGE: RACE HUD
# -------------------------------------------------
if page == "Race HUD":
    st.title("🏁 GR Strategist Lite — Race HUD (Esports Mode)")
    st.caption(f"{car} • synthetic GR Cup stint • up to {MAX_LAPS} laps")

    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Current Lap", current_lap)
    top2.metric("Last Lap Time (s)", f"{current_lap_time:.3f}")
    top3.metric("Tire Remaining (%)", f"{current_tire:.1f}")
    top4.metric("Fuel", f"{fuel_percent:.1f}%  (~{fuel_laps_left:.1f} laps)")

    # Attack mode bar
    st.markdown("### ⚡ Driver Attack Mode")
    att_col1, att_col2 = st.columns([3, 1])
    with att_col1:
        bar_len = int(attack // 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        st.write(f"`{bar}`  {attack:.0f}/100")
    with att_col2:
        if attack > 80:
            st.markdown("#### ⚡⚡ ATTACK!")
        elif attack > 60:
            st.markdown("#### 🔺 Push")
        else:
            st.markdown("#### 🟦 Manage")

    st.markdown("---")
    st.subheader("🛞 Angled GR86 Tire HUD (psi / temp / health)")

    # Compute simple "health" per tire from global tire remaining & temps
    def tire_health(temp: float) -> float:
        overheat_penalty = max(0, temp - 105) * 0.7
        return float(np.clip(current_tire - overheat_penalty, 0, 100))

    fl_health = tire_health(current_row["fl_temp"])
    fr_health = tire_health(current_row["fr_temp"])
    rl_health = tire_health(current_row["rl_temp"])
    rr_health = tire_health(current_row["rr_temp"])

    # Top tires (fronts)
    ft1, mid_car, ft2 = st.columns([3, 4, 3])
    with ft1:
        st.markdown("**FL Tire**")
        st.write(f"PSI: {current_row['fl_psi']:.1f}")
        st.write(f"Temp: {current_row['fl_temp']:.1f}°C")
        st.write(f"Health: {colored_value(fl_health, low_ok=False, unit='%')}")
    with mid_car:
        st.markdown(
            """
            <div style="text-align:center; font-size: 20px;">
            🏎️ <b>GR86 — 3D Attack View</b><br/>
            <span style="font-size:12px; color:#999;">Front-biased aero & braking load</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ft2:
        st.markdown("**FR Tire**")
        st.write(f"PSI: {current_row['fr_psi']:.1f}")
        st.write(f"Temp: {current_row['fr_temp']:.1f}°C")
        st.write(f"Health: {colored_value(fr_health, low_ok=False, unit='%')}")

    # Rear tires
    rb1, _, rb2 = st.columns([3, 4, 3])
    with rb1:
        st.markdown("**RL Tire**")
        st.write(f"PSI: {current_row['rl_psi']:.1f}")
        st.write(f"Temp: {current_row['rl_temp']:.1f}°C")
        st.write(f"Health: {colored_value(rl_health, low_ok=False, unit='%')}")
    with rb2:
        st.markdown("**RR Tire**")
        st.write(f"PSI: {current_row['rr_psi']:.1f}")
        st.write(f"Temp: {current_row['rr_temp']:.1f}°C")
        st.write(f"Health: {colored_value(rr_health, low_ok=False, unit='%')}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Lap Time Trend")
        st.line_chart(hist.set_index("lap")["lap_time"])
    with c2:
        st.subheader("Tire Wear vs Laps")
        st.line_chart(hist.set_index("lap")["tire_remaining"])

# -------------------------------------------------
# PAGE: STRATEGY & PIT WINDOW
# -------------------------------------------------
elif page == "Strategy & Pit Window":
    st.title("🧠 Strategy & Pit Window")
    st.caption(f"{car} • stint management")

    st.metric("Current Lap", current_lap)
    st.metric("Estimated fuel laps left", f"{fuel_laps_left:.1f}")

    if pit_start is None:
        st.info("Not enough forecast to recommend a pit window yet.")
    else:
        st.write(
            f"🔍 **Recommended pit window:** laps **{pit_start}–{pit_end}** "
            f"(tire 30–45%, fuel safe)."
        )
        if uo:
            gain = uo["gain_seconds"]
            early = uo["early_lap"]
            late = uo["late_lap"]
            if gain > 0:
                st.success(
                    f"📈 **Undercut advantage:** Pitting on lap {early} "
                    f"is ~**{gain:.1f}s faster** over the next stint than staying out until lap {late}."
                )
            elif gain < 0:
                st.warning(
                    f"📉 **Overcut advantage:** Staying out until lap {late} "
                    f"is ~**{abs(gain):.1f}s faster** than pitting early on lap {early}."
                )
            else:
                st.info("⚖ Under vs overcut are roughly equivalent over the next stint.")

    with st.expander("Show future prediction table"):
        if not future.empty:
            st.dataframe(future.reset_index(drop=True))
        else:
            st.write("No future data yet.")

# -------------------------------------------------
# PAGE: TELEMETRY & DRIVER
# -------------------------------------------------
elif page == "Telemetry & Driver":
    st.title("📡 Telemetry & Driver Behaviour")
    st.caption(f"{car} • driver inputs & consistency")

    st.write(f"**Driver consistency:** {cons_score}/100 – {cons_text}")

    tcol1, tcol2, tcol3 = st.columns(3)
    tcol1.metric("Throttle Avg (%)", f"{current_row['throttle_pct']:.0f}")
    tcol2.metric("Brake Avg (%)", f"{current_row['brake_pct']:.0f}")
    tcol3.metric("G-Forces Lat/Long", f"{current_row['g_lat']:.1f} / {current_row['g_long']:.1f}")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Throttle vs Laps")
        st.line_chart(hist.set_index("lap")["throttle_pct"])
    with c2:
        st.subheader("Brake vs Laps")
        st.line_chart(hist.set_index("lap")["brake_pct"])

    with st.expander("Raw lap data"):
        st.dataframe(hist.reset_index(drop=True))

# -------------------------------------------------
# PAGE: TRACK & WEATHER
# -------------------------------------------------
elif page == "Track & Weather":
    st.title("🌦 Track & Weather Impact (Prototype)")
    st.caption("Simulated relationship between ambient temp, tire wear and lap time.")

    # Simple synthetic weather model
    laps = hist["lap"].values
    ambient = 22 + 0.12 * laps  # warm-up during the race
    track_temp = ambient + 8

    weather_df = pd.DataFrame(
        {
            "lap": laps,
            "ambient_C": ambient,
            "track_C": track_temp,
            "lap_time": hist["lap_time"].values,
            "tire_remaining": hist["tire_remaining"].values,
        }
    ).set_index("lap")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Track Temperature vs Lap")
        st.line_chart(weather_df[["ambient_C", "track_C"]])
    with c2:
        st.subheader("Lap Time vs Track Temp")
        st.line_chart(weather_df[["lap_time", "tire_remaining"]])

    st.write(
        """
        **Interpretation (for judges):**
        - As track temp rises, tires overheat → wear accelerates.
        - The tool can be extended to:
          * Suggest earlier pit windows on hot races.
          * Recommend compound changes (Soft/Medium) when temps drop.
        """
    )

# -------------------------------------------------
# PAGE: PIT LOSS TOOL
# -------------------------------------------------
elif page == "Pit Loss Tool":
    st.title("⛽ Pit Stop Loss & Risk Tool (Prototype)")
    st.caption("Simple model to estimate total loss for a pit stop at different laps.")

    in_lap_loss = st.slider("In-lap time loss (sec)", 4.0, 15.0, 8.0, 0.5)
    pit_lane_loss = st.slider("Pit lane time (sec)", 15.0, 30.0, 22.0, 0.5)
    out_lap_loss = st.slider("Out-lap time loss (sec)", 2.0, 12.0, 6.0, 0.5)

    total_pit_loss = in_lap_loss + pit_lane_loss + out_lap_loss
    st.metric("Estimated total pit loss", f"{total_pit_loss:.1f} sec")

    st.write(
        """
        **How a team would use this:**
        - Combine pit loss with undercut/overcut prediction
        - Decide if the time gained on fresh tyres is worth the stop now
        - Combine with safety car probability (future extension)
        """
    )

# -------------------------------------------------
# TRACK STRESS MAP (SHARED)
# -------------------------------------------------
st.markdown("---")
st.subheader("🔥 Track Sector Stress Map (shared across pages)")

s1, s2, s3 = sector_heat(hist)
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
