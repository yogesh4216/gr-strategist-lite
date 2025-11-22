import streamlit as st
import numpy as np
import pandas as pd
import time

st.set_page_config(page_title="GR Strategist Lite 🏎️", layout="wide")

st.title("🏁 GR Strategist Lite — Real-Time Pit Strategy Assistant")

# Car dropdown
st.sidebar.header("Car Selection")
selected_car = st.sidebar.selectbox("Choose Car:", ["Car 01", "Car 02", "Car 03"])

# Initialize session state for telemetry
if "lap" not in st.session_state:
    st.session_state.lap = 1
if "pace" not in st.session_state:
    st.session_state.pace = []
if "tire_wear" not in st.session_state:
    st.session_state.tire_wear = []

# Simulated telemetry values
pace_drop = min(60, st.session_state.lap * 0.25)  # 0.25s per lap degradation
current_pace = 100 - pace_drop

st.session_state.pace.append(current_pace)
tire_remaining = max(0, 100 - (st.session_state.lap * 4))  # 4% per lap wear
st.session_state.tire_wear.append(tire_remaining)

# Display telemetry
st.subheader(f"📡 Live Telemetry — {selected_car}")
col1, col2, col3 = st.columns(3)

col1.metric("Current Lap", st.session_state.lap)
col2.metric("Predicted Pace (s)", f"{current_pace:.2f}")
col3.metric("Tire % Remaining", f"{tire_remaining:.0f}%")

# Charts
col_a, col_b = st.columns(2)
with col_a:
    st.line_chart(st.session_state.pace, height=250)
    st.caption("Lap pace trend (lower = faster)")

with col_b:
    st.line_chart(st.session_state.tire_wear, height=250)
    st.caption("Tire health degradation")

# Strategy rule
if tire_remaining < 40:
    st.success("🟢 PIT NOW — Tire below 40%!")
elif st.session_state.lap > 12:
    st.warning("🟡 Consider pitting in next laps")
else:
    st.info("🔵 Good pace — Stay out!")

# Auto increment lap
st.session_state.lap += 1
