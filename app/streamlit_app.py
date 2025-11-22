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

FUEL_STINT_LAPS = 20
FUEL_PERCENT_PER_LAP = 100.0 / FUEL_STINT_LAPS

BASE_PRESSURE_PSI = 24.0
PRESSURE_NOISE = 0.7

TEMP_BASE = 80.0           # °C
TEMP_WEAR_GAIN = 0.5
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
            "is_sc_lap",
        ]
    )
    st.session_state.current_lap = 0
    st.session_state.tire_remaining = 100.0
    st.session_state.fuel_percent = 100.0

# safety car state
if "sc_active" not in st.session_state:
    st.session_state.sc_active = False
if "sc_laps_remaining" not in st.session_state:
    st.session_state.sc_laps_remaining = 0


# -------------------------------------------------
# ANALYTICS HELPERS
# -------------------------------------------------
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
    base = 0.5 * (row["throttle_pct"] / 100) + 0.3 * (row["brake_pct"] / 100)
    base += 0.2 * (row["g_lat"] / 2.0)
    return float(np.clip(base * 100, 0, 100))


def safety_car_probability(row: pd.Series) -> float:
    """
    Simple safety car risk model (0–1):
    - High tire wear
    - Low fuel
    - High G-forces
    - High aggression
    """
    risk = 0.0

    # Tire-based risk
    if row["tire_remaining"] < 40:
        risk += (40 - row["tire_remaining"]) * 0.015

    # Fuel-based risk
    if row["fuel_percent"] < 20:
        risk += (20 - row["fuel_percent"]) * 0.02

    # G-force risk (off-track, spins)
    if row["g_lat"] > 2.0 or row["g_long"] > 1.8:
        risk += 0.15

    # Aggression risk
    aggr = aggression_level(row)
    if aggr > 80:
        risk += 0.12
    elif aggr > 65:
        risk += 0.06

    return float(np.clip(risk, 0.0, 1.0))


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

    if (pit_start is not None) and (pit_end is not None):
        if pit_start <= current_lap <= pit_end:
            messages.append((
                2, "warn", "Inside ideal pit window",
                f"🧠 You are inside the recommended pit window (laps {pit_start}-{pit_end}). "
                "Fresh tyres now will stabilise pace."
            ))

    if not messages:
        return ("ok", "Stay out", "✅ Tires and fuel are within a comfortable range. Stay out and manage pace.")

    messages.sort(key=lambda x: x[0], reverse=True)
    _, level, title, msg = messages[0]
    return (level, title, msg)


# -------------------------------------------------
# TRACK MAP HELPERS (Road America schematic)
# -------------------------------------------------
def get_track_polyline(gps_mode: bool):
    """
    Returns a schematic polyline of Road America in normalized coordinates (0–100).
    gps_mode=True is a slightly more 'GPS-like' layout – later can be replaced
    with real lat/long from VBOX data.
    """
    # Basic hand-tuned outline – shaped roughly like Road America
    base_pts = [
        (10, 10),  (30, 8),  (55, 10), (75, 20),  # main straight + T1/T3
        (85, 30),  (88, 45), (80, 60), (65, 70),  # carousel area
        (45, 78),  (30, 72), (20, 60), (15, 45),  # kink and mid-field
        (12, 30),  (10, 18), (10, 10),            # back to start/finish
    ]

    if gps_mode:
        # Slightly stretched vertically to look more like lat/long projection
        return [(x, 5 + 1.15 * (y - 5)) for (x, y) in base_pts]
    else:
        return base_pts


def build_track_map_html(view_lap: int,
                         view_row: pd.Series,
                         gps_mode: bool,
                         s1: float,
                         s2: float,
                         s3: float) -> str:
    """
    Builds an inline SVG mini-map of Road America:
    - Polyline track
    - Pit lane entry/exit (fixed region near start/finish)
    - F1-style car dot with number "86"
    - SC incident marker (⚠) on highest-stress sector when SC is active
    """
    pts = get_track_polyline(gps_mode)
    n_pts = len(pts)

    # Map lap -> index along track (playback "replay" around the lap)
    car_idx = (view_lap - 1) % n_pts
    car_x, car_y = pts[car_idx]

    # Build polyline points string
    poly_str = " ".join(f"{x},{y}" for x, y in pts)

    # Sector indices for incident placement (rough mapping)
    sector_map = {
        0: int(n_pts * 0.20),  # S1
        1: int(n_pts * 0.55),  # S2
        2: int(n_pts * 0.80),  # S3
    }

    # Incident marker if this is a Safety Car lap
    incident_svg = ""
    if bool(view_row.get("is_sc_lap", False)):
        stresses = np.array([s1, s2, s3])
        sec_idx = int(np.argmax(stresses))
        inc_idx = sector_map.get(sec_idx, car_idx)
        ix, iy = pts[inc_idx]
        incident_svg = f"""
      <g>
        <circle cx="{ix}" cy="{iy}" r="3" fill="#FFEB3B" stroke="#000" stroke-width="0.5"></circle>
        <text x="{ix}" y="{iy+1.4}" text-anchor="middle"
              font-size="3.2" font-weight="bold" fill="#000">⚠</text>
      </g>
"""

    # Pit lane region – static box near start/finish (future: distance-based)
    pit_x1, pit_y1 = 6, 6
    pit_w, pit_h = 10, 6

    gps_label = "GPS layout (normalized lat/long)" if gps_mode else "Schematic circuit layout"

    svg = f"""
<div style="margin-top:8px; display:flex; flex-direction:column; align-items:center;">
  <div style="font-size:13px;color:#bbb;margin-bottom:6px;">
    Road America – Live Car Tracker • {gps_label}
  </div>
  <svg width="520" height="320" viewBox="0 0 100 80" style="background:#05060B;border-radius:12px;border:1px solid #222;">
    <!-- Track outline -->
    <polyline points="{poly_str}"
              fill="none"
              stroke="#888"
              stroke-width="1.4"
              stroke-linecap="round"
              stroke-linejoin="round" />

    <!-- Pit lane (entry/exit near start/finish) -->
    <rect x="{pit_x1}" y="{pit_y1}" width="{pit_w}" height="{pit_h}"
          fill="rgba(255,255,255,0.03)" stroke="#FF9100" stroke-width="0.7" />
    <text x="{pit_x1 + pit_w/2}" y="{pit_y1 - 1.5}" text-anchor="middle"
          font-size="3" fill="#FFB74D">PIT</text>

    <!-- Car marker: F1-style dot with number 86 -->
    <circle cx="{car_x}" cy="{car_y}" r="2.6"
            fill="#00E676" stroke="#FFFFFF" stroke-width="0.6" />
    <text x="{car_x}" y="{car_y + 1.0}" text-anchor="middle"
          font-size="3.0" fill="#111" font-weight="bold">86</text>

    <!-- Start/finish line -->
    <line x1="10" y1="9" x2="10" y2="13" stroke="#FFFFFF" stroke-width="0.7" />
    <text x="12" y="15" font-size="2.8" fill="#ccc">S/F</text>

    <!-- Incident marker (SC) -->
    {incident_svg}
  </svg>
  <div style="font-size:11px;color:#777;margin-top:4px;">
    Dot moves around the circuit as you scrub laps • Incident marker shows likely SC zone.
  </div>
</div>
"""
    return svg


# -------------------------------------------------
# SIMULATION HELPERS (include Safety Car logic)
# -------------------------------------------------
def simulate_next_lap() -> dict:
    """Generate the next lap with realistic-looking telemetry."""
    prev_lap = st.session_state.current_lap
    lap = prev_lap + 1

    # --- Auto safety car trigger (A + B) ---
    if (not st.session_state.sc_active) and len(st.session_state.history) > 0:
        prev_row = st.session_state.history.iloc[-1]
        risk = safety_car_probability(prev_row)  # B: risk-based
        base_chance = 0.02                       # A: small random base
        chance = base_chance + risk * 0.25
        if RNG.random() < chance:
            st.session_state.sc_active = True
            st.session_state.sc_laps_remaining = int(RNG.integers(2, 5))

    sc_active = st.session_state.sc_active

    wear_variation = RNG.normal(0, 0.8)

    # Under SC: softer wear & fuel, slower lap
    wear_per_lap = WEAR_PER_LAP * (0.35 if sc_active else 1.0)
    fuel_per_lap = FUEL_PERCENT_PER_LAP * (0.6 if sc_active else 1.0)

    tire_remaining = max(
        0.0, st.session_state.tire_remaining - wear_per_lap + wear_variation
    )

    fuel_percent = max(
        0.0, st.session_state.fuel_percent - fuel_per_lap
    )

    if sc_active:
        throttle_mean = 0.6
        brake_mean = 0.25
        g_lat_mean = 1.0
        g_long_mean = 0.8
        noise_scale = 0.4
        sc_time_delta = 18.0
    else:
        throttle_mean = 0.75
        brake_mean = 0.35
        g_lat_mean = 1.4
        g_long_mean = 1.1
        noise_scale = 0.5
        sc_time_delta = 0.0

    throttle = np.clip(RNG.normal(throttle_mean, 0.1), 0.3, 1.0)
    brake = np.clip(RNG.normal(brake_mean, 0.1), 0.05, 1.0)

    degradation = (100.0 - tire_remaining) * PACE_PER_WEAR
    smoothness_factor = (1.1 - throttle + 0.3 * brake)
    noise = RNG.normal(0, noise_scale)
    lap_time = BASE_PACE + degradation * smoothness_factor + noise + sc_time_delta

    base_temp = TEMP_BASE + (100.0 - tire_remaining) * TEMP_WEAR_GAIN
    fl_temp = base_temp + RNG.normal(0, TEMP_NOISE) + brake * 10
    fr_temp = base_temp + RNG.normal(0, TEMP_NOISE) + brake * 11
    rl_temp = base_temp - 5 + RNG.normal(0, TEMP_NOISE) + throttle * 4
    rr_temp = base_temp - 3 + RNG.normal(0, TEMP_NOISE) + throttle * 5

    fl_psi = BASE_PRESSURE_PSI + RNG.normal(0, PRESSURE_NOISE) + (fl_temp - TEMP_BASE) * 0.02
    fr_psi = BASE_PRESSURE_PSI + RNG.normal(0, PRESSURE_NOISE) + (fr_temp - TEMP_BASE) * 0.02
    rl_psi = BASE_PRESSURE_PSI - 0.5 + RNG.normal(0, PRESSURE_NOISE) + (rl_temp - TEMP_BASE) * 0.015
    rr_psi = BASE_PRESSURE_PSI - 0.3 + RNG.normal(0, PRESSURE_NOISE) + (rr_temp - TEMP_BASE) * 0.015

    g_lat = np.clip(RNG.normal(g_lat_mean, 0.15), 0.8, 2.5)
    g_long = np.clip(RNG.normal(g_long_mean, 0.12), 0.6, 2.0)

    s1_stress = np.clip(0.4 + brake * 0.8 + (100 - tire_remaining) / 200, 0, 1)
    s2_stress = np.clip(0.3 + throttle * 0.5 + g_lat / 4, 0, 1)
    s3_stress = np.clip(0.5 + throttle * 0.6 + (100 - tire_remaining) / 230, 0, 1)

    # SC countdown
    if sc_active:
        st.session_state.sc_laps_remaining -= 1
        if st.session_state.sc_laps_remaining <= 0:
            st.session_state.sc_active = False

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
        is_sc_lap=sc_active,
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


# -------------------------------------------------
# SIDEBAR NAV + MANUAL SC CONTROL (C)
# -------------------------------------------------
st.sidebar.title("GR Strategist Lite")
car = st.sidebar.selectbox("Car", ["GR86-035 Demo", "GR86-047 Demo"])
target_lap = st.sidebar.slider("Simulate laps up to", 5, MAX_LAPS, 15)

page = st.sidebar.radio(
    "View",
    ["Race HUD", "Strategy & Pit Window", "Telemetry & Driver", "Track & Weather", "Pit Loss Tool"],
)

st.sidebar.markdown("### Safety Car Controls (Demo)")
if st.sidebar.button("Force SC (3 laps)"):
    st.session_state.sc_active = True
    st.session_state.sc_laps_remaining = 3
if st.sidebar.button("Clear SC"):
    st.session_state.sc_active = False
    st.session_state.sc_laps_remaining = 0

if st.sidebar.button("Reset stint"):
    st.session_state.clear()
    st.experimental_rerun()

ensure_laps_upto(target_lap)
hist = st.session_state.history.copy()

if hist.empty:
    st.info("Simulating initial laps…")
    st.stop()

# latest state (for non-playback pages)
current_row = hist.iloc[-1]
current_lap = int(current_row["lap"])
current_tire = float(current_row["tire_remaining"])
fuel_percent = float(current_row["fuel_percent"])
fuel_laps_left = fuel_percent / FUEL_PERCENT_PER_LAP if FUEL_PERCENT_PER_LAP > 0 else 0.0
current_lap_time = float(current_row["lap_time"])
attack_latest = aggression_level(current_row)

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

    # 🎥 Playback controls (lap view)
    playback_lap = st.slider("🎥 Playback lap view", 1, current_lap, current_lap)
    view_row = hist[hist["lap"] == playback_lap].iloc[0]

    view_lap = int(view_row["lap"])
    view_tire = float(view_row["tire_remaining"])
    view_fuel = float(view_row["fuel_percent"])
    view_fuel_laps_left = view_fuel / FUEL_PERCENT_PER_LAP if FUEL_PERCENT_PER_LAP > 0 else 0.0
    view_lap_time = float(view_row["lap_time"])
    view_attack = aggression_level(view_row)

    # Top metrics based on playback lap
    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Viewing Lap", view_lap, f"of {current_lap}")
    top2.metric("Lap Time (s)", f"{view_lap_time:.3f}")
    top3.metric("Tire Remaining (%)", f"{view_tire:.1f}")
    top4.metric("Fuel", f"{view_fuel:.1f}%  (~{view_fuel_laps_left:.1f} laps)")

    # AI Pit Engineer (recomputed for viewed lap)
    ai_level, ai_title, ai_message = ai_pit_advice(
        current_lap=view_lap,
        current_tire=view_tire,
        fuel_laps_left=view_fuel_laps_left,
        pit_start=pit_start,
        pit_end=pit_end,
        uo=uo,
    )

    st.markdown("### 🎧 AI Pit Engineer")
    if ai_level == "critical":
        st.error(f"**{ai_title}** – {ai_message}")
    elif ai_level == "warn":
        st.warning(f"**{ai_title}** – {ai_message}")
    else:
        st.success(f"**{ai_title}** – {ai_message}")

    # Safety car risk using viewed lap
    risk = safety_car_probability(view_row)
    st.markdown("### 🚨 Safety Car Risk Indicator")
    if bool(view_row.get("is_sc_lap", False)):
        st.error("🟡 SAFETY CAR LAP – field bunched, reduced wear & fuel.")
    if risk > 0.6:
        st.error(f"🟥 Critical | {risk*100:.1f}% — SC deployment likely, reduce curb load & aggression!")
    elif risk > 0.35:
        st.warning(f"🟧 Elevated | {risk*100:.1f}% — watch tire integrity, fuel and track limits.")
    else:
        st.success(f"🟩 Low | {risk*100:.1f}% — stable phase, no SC expected soon.")

    # Driver attack mode
    st.markdown("### ⚡ Driver Attack Mode")
    att_col1, att_col2 = st.columns([4, 1])
    with att_col1:
        bar_len = int(view_attack // 5)           # 0–20 blocks
        green_len = min(bar_len, 12)
        yellow_len = max(min(bar_len - 12, 4), 0)
        red_len = max(bar_len - 16, 0)

        bar = (
            "🟩" * green_len +
            "🟨" * yellow_len +
            "🟥" * red_len +
            "▫️" * (20 - bar_len)
        )
        st.write(f"`{bar}`  **{view_attack:.0f}/100**")

    with att_col2:
        if view_attack > 80:
            st.markdown("#### ⚡⚡ ATTACK!")
        elif view_attack > 60:
            st.markdown("#### 🔺 Push")
        else:
            st.markdown("#### 🟦 Manage")

    st.markdown("---")
    st.subheader("🚗 Top-Down GR86 Tire Heat Map (psi / temp / health)")

    def tire_health(temp: float) -> float:
        overheat_penalty = max(0, temp - 105) * 0.7
        return float(np.clip(view_tire - overheat_penalty, 0, 100))

    def tire_color(temp: float) -> str:
        if temp < 85:
            return "#2196F3"
        elif temp < 100:
            return "#00E676"
        elif temp < 115:
            return "#FF9100"
        else:
            return "#FF1744"

    fl_temp = float(view_row["fl_temp"])
    fr_temp = float(view_row["fr_temp"])
    rl_temp = float(view_row["rl_temp"])
    rr_temp = float(view_row["rr_temp"])

    fl_psi = float(view_row["fl_psi"])
    fr_psi = float(view_row["fr_psi"])
    rl_psi = float(view_row["rl_psi"])
    rr_psi = float(view_row["rr_psi"])

    fl_health = tire_health(fl_temp)
    fr_health = tire_health(fr_temp)
    rl_health = tire_health(rl_temp)
    rr_health = tire_health(rr_temp)

    fl_col = tire_color(fl_temp)
    fr_col = tire_color(fr_temp)
    rl_col = tire_color(rl_temp)
    rr_col = tire_color(rr_temp)

    def tire_html(label: str, temp: float, psi: float, health: float, color: str) -> str:
        tooltip = f"{label}: {temp:.1f}°C • {psi:.1f} psi • {health:.1f}% health"
        # all HTML kept inside this string so nothing leaks
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

    heatmap_html = f"""
<div style="display:flex;flex-direction:column;align-items:center;gap:18px;margin-top:8px;">
  <div style="font-size:13px;color:#bbb;">FRONT AXLE</div>
  <div style="display:flex;gap:70px;align-items:center;justify-content:center;">
    {tire_html("FL", fl_temp, fl_psi, fl_health, fl_col)}
    {car_body_html}
    {tire_html("FR", fr_temp, fr_psi, fr_health, fr_col)}
  </div>

  <div style="font-size:13px;color:#bbb;margin-top:10px;">REAR AXLE</div>
  <div style="display:flex;gap:220px;align-items:center;justify-content:center;">
    {tire_html("RL", rl_temp, rl_psi, rl_health, rl_col)}
    {tire_html("RR", rr_temp, rr_psi, rr_health, rr_col)}
  </div>
</div>
"""

    st.markdown(heatmap_html, unsafe_allow_html=True)

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

        # ================================
    # ROAD AMERICA TRACK MAP SECTION
    # ================================
    import random
st.markdown("### 🗺 Race Map View (Prototype)")
gps_mode = st.checkbox("Enable GPS Layout Mode", value=True)

# Build improved track map
track_html = build_track_map_html(
    view_lap=view_lap,
    view_row=view_row,
    gps_mode=gps_mode,
    s1=s1,
    s2=s2,
    s3=s3,
)

st.markdown(track_html, unsafe_allow_html=True)

# -------------------------------------------------
# STRATEGY & PIT WINDOW
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
# TELEMETRY & DRIVER
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
# TRACK & WEATHER
# -------------------------------------------------
elif page == "Track & Weather":
    st.title("🌦 Track & Weather Impact (Prototype)")
    st.caption("Simulated relationship between ambient temp, tire wear and lap time.")

    laps = hist["lap"].values
    ambient = 22 + 0.12 * laps
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
# PIT LOSS TOOL
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