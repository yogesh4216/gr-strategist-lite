import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="GR Strategist Lite 🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------
# CONFIG
# ----------------------
MAX_LAPS = 25
BASE_PACE = 100.0          # seconds
WEAR_PER_LAP = 4.0         # % tire lost per lap
PACE_PER_WEAR = 0.25       # sec of pace loss per 1% tire lost


# ----------------------
# STATE INIT
# ----------------------
if "history" not in st.session_state:
    # history dataframe: one row per completed lap
    st.session_state.history = pd.DataFrame(
        columns=[
            "lap",
            "lap_time",
            "pace",
            "tire_remaining",
            "s1_stress",
            "s2_stress",
            "s3_stress",
        ]
    )
    st.session_state.current_lap = 0
    st.session_state.tire_remaining = 100.0


# ----------------------
# HELPERS
# ----------------------
rng = np.random.default_rng(42)


def simulate_next_lap(prev_lap: int, tire_remaining: float) -> dict:
    """Create a synthetic but realistic next lap."""
    lap = prev_lap + 1

    # Tire wear model
    wear_noise = rng.normal(0, 0.7)
    tire_remaining = max(0.0, tire_remaining - WEAR_PER_LAP + wear_noise)

    # Pace model: slower as tire wears
    degradation = (100.0 - tire_remaining) * PACE_PER_WEAR
    noise = rng.normal(0, 0.4)
    lap_time = BASE_PACE + degradation + noise

    # Sector stress (0–1 scale) – weight corners vs straights
    s1_stress = min(1.0, 0.4 + (100 - tire_remaining) / 150 + rng.normal(0, 0.05))
    s2_stress = min(1.0, 0.3 + (100 - tire_remaining) / 170 + rng.normal(0, 0.05))
    s3_stress = min(1.0, 0.5 + (100 - tire_remaining) / 130 + rng.normal(0, 0.05))

    return dict(
        lap=lap,
        lap_time=lap_time,
        pace=lap_time,  # same here – could separate in future
        tire_remaining=tire_remaining,
        s1_stress=s1_stress,
        s2_stress=s2_stress,
        s3_stress=s3_stress,
    )


def ensure_laps_upto(n_laps: int):
    """Simulate laps up to n_laps."""
    while st.session_state.current_lap < n_laps and n_laps <= MAX_LAPS:
        row = simulate_next_lap(
            st.session_state.current_lap, st.session_state.tire_remaining
        )
        st.session_state.current_lap = int(row["lap"])
        st.session_state.tire_remaining = float(row["tire_remaining"])
        st.session_state.history = pd.concat(
            [st.session_state.history, pd.DataFrame([row])],
            ignore_index=True,
        )


def predict_future_from(current_lap: int, tire_remaining: float, last_pace: float):
    """Project laps from current_lap+1 up to MAX_LAPS."""
    laps = []
    lap = current_lap
    t_rem = tire_remaining
    pace = last_pace
    while lap < MAX_LAPS:
        lap += 1
        t_rem = max(0.0, t_rem - WEAR_PER_LAP)
        degradation = (100.0 - t_rem) * PACE_PER_WEAR
        pace = BASE_PACE + degradation
        laps.append({"lap": lap, "tire_remaining": t_rem, "pace": pace})
    return pd.DataFrame(laps)


def pick_pit_window(future_df: pd.DataFrame):
    """
    Simple pit-window heuristic:
      - target tire between 35–45%
      - choose smallest predicted pace in that region
    """
    if future_df.empty:
        return None, None

    window = future_df[
        (future_df["tire_remaining"] <= 45) & (future_df["tire_remaining"] >= 35)
    ]
    if window.empty:
        # fallback: when tire first goes below 50%
        window = future_df[future_df["tire_remaining"] <= 50]

    if window.empty:
        return None, None

    best_row = window.sort_values("pace").iloc[0]
    best_lap = int(best_row["lap"])
    return max(best_lap - 1, 1), min(best_lap + 1, MAX_LAPS)


def under_overcut_compare(current_lap: int, best_lap: int, future_df: pd.DataFrame):
    """
    Compare pitting one lap early vs one lap late relative to best_lap.
    Very simple model: assume fresh tires reset degradation.
    """
    if best_lap is None:
        return None

    early = max(best_lap - 1, current_lap + 1)
    late = min(best_lap + 1, MAX_LAPS - 1)

    def stint_cost(pit_lap: int):
        # cost over next 6 laps (pit_lap + 5)
        horizon = min(pit_lap + 5, MAX_LAPS)
        df = future_df.copy()
        df = df[df["lap"] <= horizon]
        # laps before pit: worn tires
        before = df[df["lap"] < pit_lap]["pace"].sum()
        # after pit: reset tire to 100, recompute pace curve
        laps_after = range(pit_lap, horizon + 1)
        tire = 100.0
        after = 0.0
        for _ in laps_after:
            degradation = (100.0 - tire) * PACE_PER_WEAR
            after += BASE_PACE + degradation
            tire = max(0.0, tire - WEAR_PER_LAP)
        return before + after

    early_cost = stint_cost(early)
    late_cost = stint_cost(late)
    gain = late_cost - early_cost  # positive → early is better
    return dict(early_lap=early, late_lap=late, gain_seconds=gain)


def driver_consistency_score(df: pd.DataFrame):
    """100 = perfectly consistent. Lower if lap-time variance is high."""
    if len(df) < 3:
        return 100, "Not enough laps yet"
    std = df["lap_time"].std()
    score = max(0, 100 - std * 12)  # 0–100
    if score > 85:
        text = "Super consistent – ideal for long stints ✅"
    elif score > 70:
        text = "Good consistency – small improvements possible 👍"
    elif score > 50:
        text = "Aggressive / variable – can hurt tire life ⚠️"
    else:
        text = "Very inconsistent – high risk for race pace ❌"
    return int(score), text


def sector_heat(df: pd.DataFrame):
    if df.empty:
        return [0, 0, 0]
    return [
        float(df["s1_stress"].mean()),
        float(df["s2_stress"].mean()),
        float(df["s3_stress"].mean()),
    ]


# ----------------------
# SIDEBAR
# ----------------------
st.sidebar.header("Car / Simulation Controls")
selected_car = st.sidebar.selectbox("Car", ["GR86-035 Demo", "GR86-047 Demo"])
auto_laps = st.sidebar.slider("Simulate laps up to", 5, MAX_LAPS, 15)

if st.sidebar.button("Reset race data"):
    st.session_state.history = pd.DataFrame(
        columns=st.session_state.history.columns
    )
    st.session_state.current_lap = 0
    st.session_state.tire_remaining = 100.0

# simulate up to chosen lap
ensure_laps_upto(auto_laps)
hist = st.session_state.history.copy()

if hist.empty:
    st.write("Simulating first laps…")
    st.stop()

current_lap = int(hist["lap"].max())
current_tire = float(hist.iloc[-1]["tire_remaining"])
current_pace = float(hist.iloc[-1]["lap_time"])

future = predict_future_from(current_lap, current_tire, current_pace)
pit_start, pit_end = pick_pit_window(future)
uo = under_overcut_compare(current_lap, pit_start + 1 if pit_start else None, future)
cons_score, cons_text = driver_consistency_score(hist)
s1, s2, s3 = sector_heat(hist)

# ----------------------
# TOP METRICS
# ----------------------
st.title("🏁 GR Strategist Lite — Real-Time Pit Strategy Assistant")
st.caption(f"Simulated stint • {selected_car} • Max {MAX_LAPS} laps")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Lap", current_lap)
col2.metric("Last Lap Time (s)", f"{current_pace:.2f}")
col3.metric("Tire Remaining", f"{current_tire:.1f}%")
col4.metric("Driver Consistency", f"{cons_score}/100")

st.write(f"**Consistency analysis:** {cons_text}")

# ----------------------
# MAIN CHARTS
# ----------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Lap Pace Trend (lower = faster)")
    st.line_chart(hist.set_index("lap")["lap_time"])

with c2:
    st.subheader("Tire Wear Over Stint")
    st.line_chart(hist.set_index("lap")["tire_remaining"])

# ----------------------
# PIT WINDOW & STRATEGY
# ----------------------
st.markdown("---")
st.subheader("🧠 Strategy Engine — Pit Window & Under/Overcut")

if pit_start is None:
    st.info("Not enough future data to recommend a pit window yet.")
else:
    st.write(
        f"🔍 **Recommended pit window:** laps **{pit_start}–{pit_end}** "
        f"(tire ~35–45% and pace still salvageable)"
    )

    if uo:
        gain = uo["gain_seconds"]
        early_lap = uo["early_lap"]
        late_lap = uo["late_lap"]
        if gain > 0:
            st.success(
                f"📈 **Undercut advantage:** Pitting **early on lap {early_lap}** "
                f"is estimated ~**{gain:.1f}s faster** over the next stint "
                f"than pitting later on lap {late_lap}."
            )
        elif gain < 0:
            st.warning(
                f"📉 **Overcut advantage:** Staying out until **lap {late_lap}** "
                f"is estimated ~**{abs(gain):.1f}s faster** than an early stop "
                f"on lap {early_lap}."
            )
        else:
            st.info(
                f"⚖️ Under vs overcut are neutral between laps {early_lap} and {late_lap}."
            )

# ----------------------
# TRACK HEAT MAP (SECTOR STRESS)
# ----------------------
st.markdown("---")
st.subheader("🔥 Track Sector Stress Map")

heat_df = pd.DataFrame(
    {
        "Sector": ["S1 – Braking Zone", "S2 – High Speed", "S3 – Traction"],
        "Stress": [s1, s2, s3],
    }
)

st.bar_chart(heat_df.set_index("Sector"))

labels = []
for sector, stress in zip(heat_df["Sector"], heat_df["Stress"]):
    if stress > 0.8:
        labels.append(f"🟥 {sector}: Very high stress – manage tires here.")
    elif stress > 0.6:
        labels.append(f"🟧 {sector}: Moderate–high stress – driver must be smooth.")
    else:
        labels.append(f"🟩 {sector}: OK – safe for pushing.")

for line in labels:
    st.write(line)

# ----------------------
# DATA TABLE (for nerdy judges 🤓)
# ----------------------
with st.expander("Show underlying lap data table"):
    st.dataframe(hist.reset_index(drop=True))
