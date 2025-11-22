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


def ai_pit_advice(current_lap: int,
                  current_tire: float,
                  fuel_laps_left: float,
                  pit_start,
                  pit_end,
                  uo):
    """
    Hybrid AI advisor:
    - Critical if fuel or tires are very low
    - Warning if in pit window or strong undercut
    - Otherwise, stay out & manage
    Returns (level, title, message)
    """
    messages = []

    # Fuel logic
    if fuel_laps_left <= 1.5:
        messages.append((
            3, "critical", "Fuel critical",
            f"⛽ Fuel remaining for only ~{fuel_laps_left:.1f} laps. Box this lap or risk running dry."
        ))
    elif fuel_laps_left <= 3.0:
        messages.append((
            2, "warn", "Fuel window",
            f"⛽ Fuel getting low (~{fuel_laps_left:.1f} laps left). Plan to pit in the next 1–2 laps."
        ))

    # Tire logic
    if current_tire <= 25:
        messages.append((
            3, "critical", "Tire cliff incoming",
            f"🛞 Global tire life at {current_tire:.1f}%. Expect big lap-time drop – pit now or very soon."
        ))
    elif current_tire <= 40:
        messages.append((
            2, "warn", "High tire wear",
            f"🛞 Tire life {current_tire:.1f}% – you're approaching the drop-off zone."
        ))

    # Undercut / overcut logic
    if uo is not None:
        gain = uo["gain_seconds"]
        early = uo["early_lap"]
        late = uo["late_lap"]
        if gain > 5:
            messages.append((
                2, "warn", "Strong undercut opportunity",
                f"📈 Pitting around lap {early} gains ~{gain:.1f}s vs staying to lap {late}."
            ))
        elif gain > 2:
            messages.append((
                1, "info", "Mild undercut gain",
                f"📈 Undercut on lap {early} is ~{gain:.1f}s faster than pitting later."
            ))

    # If in recommended pit window, push advice
    if (pit_start is not None) and (pit_end is not None):
        if pit_start <= current_lap <= pit_end:
            messages.append((
                2, "warn", "Inside ideal pit window",
                f"🧠 You are inside the recommended pit window (laps {pit_start}-{pit_end}). "
                "Fresh tyres now will stabilise pace."
            ))

    if not messages:
        return ("ok", "Stay out", "✅ Tires and fuel are within a comfortable range. Stay out and manage pace.")

    # Pick highest priority (3 critical > 2 warn > 1 info)
    messages.sort(key=lambda x: x[0], reverse=True)
    _, level, title, msg = messages[0]
    return (level, title, msg)


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

# Get AI pit advice (Hybrid mode)
ai_level, ai_title, ai_message = ai_pit_advice(
    current_lap=current_lap,
    current_tire=current_tire,
    fuel_laps_left=fuel_laps_left,
    pit_start=pit_start,
    pit_end=pit_end,
    uo=uo,
)

# -------------------------------------------------
# PAGE: RACE HUD (with GR86 heat map + AI pit advisor)
# -------------------------------------------------
if page == "Race HUD":
    st.title("🏁 GR Strategist Lite — Race HUD (Esports Mode)")
    st.caption(f"{car} • synthetic GR Cup stint • up to {MAX_LAPS} laps")

    # ---- Top summary metrics ----
    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Current Lap", current_lap)
    top2.metric("Last Lap Time (s)", f"{current_lap_time:.3f}")
    top3.metric("Tire Remaining (%)", f"{current_tire:.1f}")
    top4.metric("Fuel", f"{fuel_percent:.1f}%  (~{fuel_laps_left:.1f} laps)")

    # ---- AI Pit Advisor banner ----
    st.markdown("### 🎧 AI Pit Engineer")
    if ai_level == "critical":
        st.error(f"**{ai_title}** – {ai_message}")
    elif ai_level == "warn":
        st.warning(f"**{ai_title}** – {ai_message}")
    else:
        st.success(f"**{ai_title}** – {ai_message}")

    # ---- Driver attack mode (RPM bar + lightning) ----
    st.markdown("### ⚡ Driver Attack Mode")
    att_col1, att_col2 = st.columns([4, 1])
    with att_col1:
        bar_len = int(attack // 5)           # 0–20 blocks
        green_len = min(bar_len, 12)
        yellow_len = max(min(bar_len - 12, 4), 0)
        red_len = max(bar_len - 16, 0)

        bar = (
            "🟩" * green_len +
            "🟨" * yellow_len +
            "🟥" * red_len +
            "▫️" * (20 - bar_len)
        )
        st.write(f"`{bar}`  **{attack:.0f}/100**")

    with att_col2:
        if attack > 80:
            st.markdown("#### ⚡⚡ ATTACK!")
        elif attack > 60:
            st.markdown("#### 🔺 Push")
        else:
            st.markdown("#### 🟦 Manage")

    st.markdown("---")
    st.subheader("🚗 Top-Down GR86 Tire Heat Map (psi / temp / health)")

    # ---- Tire health & color helpers ----
    def tire_health(temp: float) -> float:
        # Global wear minus extra penalty for overheating
        overheat_penalty = max(0, temp - 105) * 0.7
        return float(np.clip(current_tire - overheat_penalty, 0, 100))

    def tire_color(temp: float) -> str:
        # Returns a hex color for the tire fill based on temp bands
        if temp < 85:
            return "#2196F3"   # blue (cold)
        elif temp < 100:
            return "#00E676"   # green (optimal)
        elif temp < 115:
            return "#FF9100"   # orange (hot)
        else:
            return "#FF1744"   # red (critical)

    # Current temps / pressures
    fl_temp = float(current_row["fl_temp"])
    fr_temp = float(current_row["fr_temp"])
    rl_temp = float(current_row["rl_temp"])
    rr_temp = float(current_row["rr_temp"])

    fl_psi = float(current_row["fl_psi"])
    fr_psi = float(current_row["fr_psi"])
    rl_psi = float(current_row["rl_psi"])
    rr_psi = float(current_row["rr_psi"])

    # Health values
    fl_health = tire_health(fl_temp)
    fr_health = tire_health(fr_temp)
    rl_health = tire_health(rl_temp)
    rr_health = tire_health(rr_temp)

    # Colors
    fl_col = tire_color(fl_temp)
    fr_col = tire_color(fr_temp)
    rl_col = tire_color(rl_temp)
    rr_col = tire_color(rr_temp)

    def tire_html(label: str, temp: float, psi: float, health: float, color: str) -> str:
        """One tire with tooltip (psi/temp/health)."""
        tooltip = f"{label}: {temp:.1f}°C • {psi:.1f} psi • {health:.1f}% health"
        return f"""
        <div style='text-align:center;' title="{tooltip}">
          <div style="
              width:70px;height:70px;
              border-radius:50%;
              border:3px solid #111;
              box-shadow:0 0 14px {color};
              background:radial-gradient(circle at 30% 30%,
                                         rgba(255,255,255,0.35),
                                         {color});
          "></div>
          <div style="font-size:11px;margin-top:4px;color:#ddd;">{label}</div>
          <div style="font-size:10px;color:#aaa;">{temp:.0f}°C • {psi:.1f} psi</div>
        </div>
        """

    # GR livery car body (white / red / black)
    car_body_html = """
    <div style="
        width:180px;height:90px;
        border-radius:24px;
        border:3px solid #111;
        background:
           linear-gradient(90deg,
                #ffffff 0%,
                #ffffff 40%,
                #ff0000 40%,
                #ff0000 72%,
                #000000 72%,
                #000000 100%);
        box-shadow:0 0 20px #ff174455;
        display:flex;
        align-items:center;
        justify-content:center;
        color:#f5f5f5;
        font-size:13px;
        font-weight:600;
    ">
      GR86 — Top View
    </div>
    <div style="font-size:10px;color:#aaa;margin-top:2px;text-align:center;">
      Toyota Gazoo Racing livery • front-biased aero & braking load
    </div>
    """

    # Layout: FRONT row + REAR row
    heatmap_html = f"""
    <div style="display:flex;flex-direction:column;align-items:center;gap:18px;margin-top:8px;">
      <div style="font-size:13px;color:#bbb;">FRONT AXLE</div>
      <div style="display:flex;gap:70px;align-items:center;justify-content:center;">
        {tire_html("FL", fl_temp, fl_psi, fl_health, fl_col)}
        {car_body_html}
        {tire_html("FR", fr_temp, fr_psi, fr_health, fr_col)}
      </div>

      <div style="font-size:13px;color:#bbb;margin-top:10px;">REAR AXLE</div>
      <div style="display:flex;gap:70px;align-items:center;justify-content:center;">
        {tire_html("RL", rl_temp, rl_psi, rl_health, rl_col)}
        <div style="width:180px;"></div>
        {tire_html("RR", rr_temp, rr_psi, rr_health, rr_col)}
      </div>
    </div>
    """

    st.markdown(heatmap_html, unsafe_allow_html=True)

    # ---- Embedded alerts from heat zones ----
    alerts = []
    for label, temp, health in [
        ("FL", fl_temp, fl_health),
        ("FR", fr_temp, fr_health),
        ("RL", rl_temp, rl_health),
        ("RR", rr_temp, rr_health),
    ]:
        if temp > 115:
            alerts.append(f"{label} tire CRITICAL temp ({temp:.1f}°C) – pit soon.")
        elif temp > 105:
            alerts.append(f"{label} tire overheating ({temp:.1f}°C) – manage pace.")
        if health < 35:
            alerts.append(f"{label} tire very low health ({health:.1f}%) – nearing cliff.")

    if alerts:
        st.markdown("#### ⚠ Tire & Grip Alerts")
        for msg in alerts:
            st.warning("🛞 " + msg)
    else:
        st.markdown("#### ✅ Tires in acceptable window")
        st.info("All four tires are within safe temperature and health ranges.")

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
        st.subheader("Lap Time vs Tire & Temp")
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