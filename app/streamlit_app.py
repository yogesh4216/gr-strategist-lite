import streamlit as st
import numpy as np
import pandas as pd
import time

st.set_page_config(page_title="GR Strategist Lite 🏎️", layout="wide")

st.title("🏁 GR Strategist Lite — Real-Time Pit Strategy Assistant")

st.sidebar.header("Car Selection")
selected_car = st.sidebar.selectbox("Choose Car:", ["Car 01", "Car 02", "Car 03"])

# Fake streaming lap data
if "lap" not in st.session_state:
    st.session_state.lap = 1
if "pace" not in st.session_state:
    st.session_state.pace = []

st.subheader(f"📡 Live Telemetry — {selected_car}")

pace_drop = max(0, (st.session_state.lap * 0.2))  # simple wear model
current_pace = 100 - pace_drop
st.session_state.pace.append(current_pace)

st.metric(label="Current Lap", value=st.session_state.lap)
st.metric(label="Predicted Pace", value=f"{current_pace:.2f}")

st.line_chart(st.session_state.pace)

# Simple strategy logic
if st.session_state.lap > 10:
    st.success("🟢 Recommended: PIT NOW")
else:
    st.warning("🟡 PIT LATER — Pace still good")

# advance lap each refresh
st.session_state.lap += 1
