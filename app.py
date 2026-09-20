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
    st.subheader("🎥 Hopper Animation — Continuous Discharge")

    hopper_width = 10
    hopper_height = 15

    def generate_positions(df):
        positions = []
        for _, row in df.iterrows():
            size_class = row["Particle_Size_Class"]
            y = np.random.uniform(0, hopper_height)
            if size_class == "Fine":
                x = np.random.normal(0, 1)  # center bias
                size = 30
            elif size_class == "Coarse":
                x = np.random.choice([-hopper_width/2, hopper_width/2]) + np.random.normal(0, 0.5)
                size = 80
            else:
                x = np.random.uniform(-hopper_width/2, hopper_width/2)
                size = 50
            positions.append([x, y, size_class, size])
        return positions

    positions = generate_positions(df.sample(min(200, len(df))))

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_xlim(-hopper_width/2, hopper_width/2)
    ax.set_ylim(0, hopper_height)

    # Draw hopper funnel walls
    ax.plot([-hopper_width/2, 0], [hopper_height, 0], color="black", linewidth=2)
    ax.plot([hopper_width/2, 0], [hopper_height, 0], color="black", linewidth=2)
    ax.fill_between([-hopper_width/2, hopper_width/2], hopper_height, 0, color="lightgray", alpha=0.2)

    scat = ax.scatter([], [], s=[], alpha=0.7)

    def update(frame):
        xs, ys, sizes, colors = [], [], [], []
        for pos in positions:
            x, y, cls, size = pos
            # Move downward
            y -= 0.1
            # Funnel effect: shift x toward center as y decreases
            funnel_ratio = max(0, (hopper_height - y) / hopper_height)
            if cls == "Fine":
                x *= (1 - 0.7 * funnel_ratio)
            elif cls == "Coarse":
                x *= (1 - 0.2 * funnel_ratio)
            else:
                x *= (1 - 0.5 * funnel_ratio)

            # Reset particle if it exits the funnel bottom
            if y <= 0:
                y = hopper_height
                if cls == "Fine":
                    x = np.random.normal(0, 1)
                elif cls == "Coarse":
                    x = np.random.choice([-hopper_width/2, hopper_width/2]) + np.random.normal(0, 0.5)
                else:
                    x = np.random.uniform(-hopper_width/2, hopper_width/2)

            # Update position
            pos[0], pos[1] = x, y
            xs.append(x)
            ys.append(y)
            sizes.append(size)
            colors.append("blue" if cls == "Fine" else "red" if cls == "Coarse" else "gray")

        scat.set_offsets(np.c_[xs, ys])
        scat.set_sizes(sizes)
        scat.set_color(colors)
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=200, interval=100, blit=True)
    ani.save("hopper_animation.gif", writer="pillow")

    st.image("hopper_animation.gif", caption="Particles funnel down and discharge continuously.")




# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("Coal Particle Size Classification Dashboard | Streamlit Research Prototype")
