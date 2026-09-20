import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Coal Particle Size Classification",
    page_icon="🪨",
    layout="wide"
)

st.title("🪨 Coal Particle Size Classification Dashboard")
st.caption("Coal particle-size distribution, classification, coal-quality analysis, prediction, and hopper animation")

# ============================================================
# CONSTANTS
# ============================================================

CLASS_LABELS = ["Fine", "Small", "Medium", "Large", "Coarse"]

# ============================================================
# SAMPLE DATA GENERATOR
# ============================================================

def make_sample_data(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    particle_size = np.clip(rng.lognormal(mean=np.log(3.2), sigma=0.8, size=n), 0.05, 30)
    mass = np.clip(rng.gamma(shape=2.5, scale=12, size=n), 0.5, 100)
    moisture = np.clip(7.5 - 0.06 * particle_size + rng.normal(0, 1, n), 2, 15)
    ash = np.clip(15 + 0.7 * np.log1p(particle_size) + rng.normal(0, 3, n), 5, 35)
    density = np.clip(1.35 + 0.025 * np.log1p(particle_size) + rng.normal(0, 0.06, n), 1.15, 1.90)
    calorific_value = np.clip(27 - 0.25 * ash - 0.25 * moisture + rng.normal(0, 1.2, n), 15, 32)
    return pd.DataFrame({
        "Sample_ID": [f"COAL-{i:04d}" for i in range(1, n + 1)],
        "Particle_Size_mm": np.round(particle_size, 3),
        "Mass_g": np.round(mass, 2),
        "Moisture_%": np.round(moisture, 2),
        "Density_g_cm3": np.round(density, 3),
        "Ash_%": np.round(ash, 2),
        "Calorific_Value_MJ_kg": np.round(calorific_value, 2)
    })

# ============================================================
# FILE UPLOAD
# ============================================================

st.sidebar.header("⚙️ Dashboard Controls")
uploaded_file = st.sidebar.file_uploader("Upload coal particle-size data", type=["xlsx", "xls", "csv"])

if uploaded_file is None:
    df = make_sample_data()
    st.sidebar.info("No file uploaded. Using built-in sample dataset.")
else:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

# ============================================================
# CLASSIFICATION THRESHOLDS
# ============================================================

st.sidebar.subheader("Particle-size classification")
fine_limit = st.sidebar.number_input("Fine upper limit (mm)", min_value=0.01, max_value=100.0, value=0.5, step=0.1)
small_limit = st.sidebar.number_input("Small upper limit (mm)", min_value=float(fine_limit + 0.01), max_value=100.0, value=2.0, step=0.1)
medium_limit = st.sidebar.number_input("Medium upper limit (mm)", min_value=float(small_limit + 0.01), max_value=100.0, value=5.0, step=0.1)
large_limit = st.sidebar.number_input("Large upper limit (mm)", min_value=float(medium_limit + 0.01), max_value=200.0, value=10.0, step=0.1)

# ============================================================
# CLASSIFICATION FUNCTION
# ============================================================

def classify_particles(particle_sizes, fine_limit, small_limit, medium_limit, large_limit):
    bins = [-np.inf, fine_limit, small_limit, medium_limit, large_limit, np.inf]
    return pd.cut(particle_sizes, bins=bins, labels=CLASS_LABELS, right=False, include_lowest=True)

df["Particle_Size_Class"] = classify_particles(df["Particle_Size_mm"], fine_limit, small_limit, medium_limit, large_limit)

# ============================================================
# DASHBOARD TABS
# ============================================================

overview_tab, distribution_tab, classification_tab, quality_tab, prediction_tab, animation_tab = st.tabs(
    ["📊 Overview", "📈 Size Distribution", "🏷️ Classification", "🧪 Coal Quality", "🔮 Prediction", "🎥 Hopper Animation"]
)

# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with overview_tab:
    st.subheader("Coal Particle-Size Overview")
    st.dataframe(df.head(20), use_container_width=True)

# ============================================================
# TAB 2 — DISTRIBUTION
# ============================================================

with distribution_tab:
    st.subheader("📈 Particle-Size Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["Particle_Size_mm"], bins=30)
    st.pyplot(fig)

# ============================================================
# TAB 3 — CLASSIFICATION
# ============================================================

with classification_tab:
    st.subheader("🏷️ Classification Summary")
    st.dataframe(df[["Sample_ID", "Particle_Size_mm", "Particle_Size_Class"]], use_container_width=True)

# ============================================================
# TAB 4 — QUALITY
# ============================================================

with quality_tab:
    st.subheader("🧪 Coal Quality Relationships")
    st.dataframe(df.describe(), use_container_width=True)

# ============================================================
# TAB 5 — PREDICTION
# ============================================================

with prediction_tab:
    st.subheader("🔮 Particle-Size Class Prediction")
    particle_size_input = st.number_input("Enter particle size (mm)", min_value=0.001, value=3.0, step=0.1)
    predicted_class = classify_particles([particle_size_input], fine_limit, small_limit, medium_limit, large_limit)[0]
    st.metric("Predicted Particle-Size Class", str(predicted_class))

# ============================================================
# TAB 6 — HOPPER ANIMATION
# ============================================================

with animation_tab:
    st.subheader("🎥 Fine vs Coarse Coal Flow Animation")

    # Funnel animation (same logic as before, with funnel walls + discharge)
    # ... [animation code here] ...

    # Classification thresholds
    st.markdown("**Classification Thresholds**")
    st.write(f"Fine: < {fine_limit} mm")
    st.write(f"Small: {fine_limit} – <{small_limit} mm")
    st.write(f"Medium: {small_limit} – <{medium_limit} mm")
    st.write(f"Large: {medium_limit} – <{large_limit} mm")
    st.write(f"Coarse: ≥ {large_limit} mm")

    # Percentiles
    st.markdown("**Particle Size Percentiles**")
    st.write(f"P10 = {df['Particle_Size_mm'].quantile(0.10):.2f} mm")
    st.write(f"P25 = {df['Particle_Size_mm'].quantile(0.25):.2f} mm")
    st.write(f"P50 = {df['Particle_Size_mm'].quantile(0.50):.2f} mm")
    st.write(f"P75 = {df['Particle_Size_mm'].quantile(0.75):.2f} mm")
    st.write(f"P90 = {df['Particle_Size_mm'].quantile(0.90):.2f} mm")

    # Classification summary bar
    st.markdown("**Classification Summary**")
    class_counts = df["Particle_Size_Class"].value_counts().reindex(CLASS_LABELS, fill_value=0)
    class_percentages = class_counts / len(df) * 100
    st.bar_chart(class_percentages)

    # Scatter plots
    st.markdown("**Coal Quality Relationships**")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0,0].scatter(df["Particle_Size_mm"], df["Ash_%"], alpha=0.5)
    axes[0,0].set_title("Particle Size vs Ash")
    axes[0,1].scatter(df["Particle_Size_mm"], df["Moisture_%"], alpha=0.5)
    axes[0,1].set_title("Particle Size vs Moisture")
    axes[1,0].scatter(df["Particle_Size_mm"], df["Density_g_cm3"], alpha=0.5)
    axes[1,0].set_title("Particle Size vs Density")
    axes[1,1].scatter(df["Particle_Size_mm"], df["Calorific_Value_MJ_kg"], alpha=0.5)
    axes[1,1].set_title("Particle Size vs Calorific Value")
    st.pyplot(fig)






# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("Coal Particle Size Classification Dashboard | Streamlit Research Prototype")
