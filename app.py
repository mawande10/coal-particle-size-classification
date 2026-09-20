from pathlib import Path
import textwrap
import os

base = Path("/mnt/data/coal_150000_jan_jun_2026")
base.mkdir(parents=True, exist_ok=True)

app_code = r'''import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

st.set_page_config(
    page_title="Coal Particle Size Classification Dashboard",
    page_icon="🪨",
    layout="wide"
)

st.title("🪨 Coal Particle Size Classification Dashboard")
st.caption(
    "150,000 simulated coal particles | January–June 2026 | "
    "Monday–Friday | 08:00–16:30"
)

CLASS_LABELS = ["Fine", "Small", "Medium", "Large", "Coarse"]


# -------------------------------------------------------------------
# DATA
# -------------------------------------------------------------------
@st.cache_data
def make_sample_data(n=150_000, seed=42):
    rng = np.random.default_rng(seed)

    # 129 weekdays between 1 Jan and 30 Jun 2026.
    workdays = pd.bdate_range("2026-01-01", "2026-06-30")

    # Create timestamps only during 08:00–16:30.
    day_index = rng.integers(0, len(workdays), n)
    seconds_from_start = rng.integers(0, 8 * 60 * 60 + 30 * 60, n)

    dates = workdays[day_index]
    timestamps = (
        pd.to_datetime(dates)
        + pd.to_timedelta(seconds_from_start, unit="s")
    )

    # Coal particle-size distribution.
    particle_size = np.clip(
        rng.lognormal(mean=np.log(4.7), sigma=0.72, size=n),
        0.08,
        40.0
    )

    mass = np.clip(
        0.015 * particle_size ** 2.15 * rng.lognormal(0, 0.35, n),
        0.01,
        250
    )

    moisture = np.clip(
        12.0 - 0.08 * particle_size + rng.normal(0, 1.7, n),
        3,
        22
    )

    ash = np.clip(
        28 - 0.28 * particle_size + rng.normal(0, 5.0, n),
        6,
        65
    )

    density = np.clip(
        1.25 + 0.012 * particle_size + rng.normal(0, 0.055, n),
        1.05,
        1.75
    )

    calorific_value = np.clip(
        29 - 0.30 * ash - 0.22 * moisture
        + 0.03 * particle_size
        + rng.normal(0, 1.2, n),
        10,
        31
    )

    velocity = np.clip(
        0.9 + 0.06 * particle_size + rng.normal(0, 0.15, n),
        0.2,
        4.0
    )

    df = pd.DataFrame({
        "Particle_ID": [
            f"COAL-{i:06d}" for i in range(1, n + 1)
        ],
        "Timestamp": timestamps,
        "Date": timestamps.date,
        "Time": timestamps.strftime("%H:%M:%S"),
        "Particle_Size_mm": np.round(particle_size, 3),
        "Mass_g": np.round(mass, 3),
        "Moisture_%": np.round(moisture, 2),
        "Density_g_cm3": np.round(density, 3),
        "Ash_%": np.round(ash, 2),
        "Calorific_Value_MJ_kg": np.round(calorific_value, 2),
        "Velocity_m_s": np.round(velocity, 3)
    })

    return df


@st.cache_data
def read_uploaded_file(file_bytes, filename):
    if filename.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    return pd.read_excel(io.BytesIO(file_bytes))


def standardise_columns(df):
    df = df.copy()
    df.columns = [
        str(c).strip().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]

    aliases = {
        "Particle_Size": "Particle_Size_mm",
        "Size_mm": "Particle_Size_mm",
        "ParticleSize_mm": "Particle_Size_mm",
        "ParticleSize": "Particle_Size_mm",
        "Mass": "Mass_g",
        "Moisture": "Moisture_%",
        "Ash": "Ash_%",
        "Density": "Density_g_cm3",
        "Calorific_Value": "Calorific_Value_MJ_kg",
        "CV_MJ_kg": "Calorific_Value_MJ_kg"
    }

    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    return df


def classify_particles(values, fine, small, medium, large):
    bins = [-np.inf, fine, small, medium, large, np.inf]
    return pd.cut(
        values,
        bins=bins,
        labels=CLASS_LABELS,
        right=False,
        include_lowest=True
    )


# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------
st.sidebar.header("⚙️ Dashboard Controls")

uploaded = st.sidebar.file_uploader(
    "Upload coal particle data",
    type=["csv", "xlsx", "xls"]
)

if uploaded is None:
    df = make_sample_data()
    st.sidebar.success(
        "Using built-in 150,000-particle dataset."
    )
else:
    try:
        df = read_uploaded_file(
            uploaded.getvalue(),
            uploaded.name
        )
        df = standardise_columns(df)
        st.sidebar.success(f"Loaded: {uploaded.name}")
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        st.stop()

if "Particle_Size_mm" not in df.columns:
    st.error(
        "The dataset must contain Particle_Size_mm "
        "(or Particle_Size / Size_mm)."
    )
    st.stop()

df["Particle_Size_mm"] = pd.to_numeric(
    df["Particle_Size_mm"],
    errors="coerce"
)
df = df.dropna(subset=["Particle_Size_mm"]).copy()

if df.empty:
    st.error("No valid particle-size records were found.")
    st.stop()

st.sidebar.subheader("Particle-size classification")

fine_limit = st.sidebar.number_input(
    "Fine upper limit (mm)",
    min_value=0.01,
    max_value=100.0,
    value=0.5,
    step=0.1
)

small_limit = st.sidebar.number_input(
    "Small upper limit (mm)",
    min_value=float(fine_limit + 0.01),
    max_value=100.0,
    value=2.0,
    step=0.1
)

medium_limit = st.sidebar.number_input(
    "Medium upper limit (mm)",
    min_value=float(small_limit + 0.01),
    max_value=100.0,
    value=5.0,
    step=0.1
)

large_limit = st.sidebar.number_input(
    "Large upper limit (mm)",
    min_value=float(medium_limit + 0.01),
    max_value=200.0,
    value=10.0,
    step=0.1
)

df["Particle_Size_Class"] = classify_particles(
    df["Particle_Size_mm"],
    fine_limit,
    small_limit,
    medium_limit,
    large_limit
)

# -------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------
tabs = st.tabs([
    "📊 Overview",
    "📈 Size Distribution",
    "🏷️ Classification",
    "🧪 Coal Quality",
    "🔮 Prediction",
    "🎬 Hopper Animation"
])

overview_tab, distribution_tab, classification_tab, quality_tab, prediction_tab, hopper_tab = tabs


# -------------------------------------------------------------------
# OVERVIEW
# -------------------------------------------------------------------
with overview_tab:
    st.subheader("Coal Particle-Size Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Particles", f"{len(df):,}")
    c2.metric("Mean Size", f"{df['Particle_Size_mm'].mean():.2f} mm")
    c3.metric("Median Size", f"{df['Particle_Size_mm'].median():.2f} mm")
    c4.metric("Minimum", f"{df['Particle_Size_mm'].min():.2f} mm")
    c5.metric("Maximum", f"{df['Particle_Size_mm'].max():.2f} mm")

    counts = (
        df["Particle_Size_Class"]
        .value_counts()
        .reindex(CLASS_LABELS, fill_value=0)
    )

    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(CLASS_LABELS, counts.values)
        ax.set_xlabel("Particle-size class")
        ax.set_ylabel("Number of particles")
        ax.set_title("Coal Particle-size Classification")
        ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        summary = pd.DataFrame({
            "Class": CLASS_LABELS,
            "Particles": counts.values,
            "Percentage_%": np.round(
                counts.values / len(df) * 100, 2
            )
        })
        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

    st.subheader("Dataset preview")
    st.dataframe(
        df.head(25),
        use_container_width=True,
        hide_index=True
    )


# -------------------------------------------------------------------
# SIZE DISTRIBUTION
# -------------------------------------------------------------------
with distribution_tab:
    st.subheader("📈 Particle-Size Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["Particle_Size_mm"], bins=50)
    ax.set_xlabel("Particle size (mm)")
    ax.set_ylabel("Frequency")
    ax.set_title("Coal Particle-Size Distribution")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    stats = df["Particle_Size_mm"].describe().to_frame(
        "Particle_Size_mm"
    )

    for p in [0.10, 0.25, 0.50, 0.75, 0.90]:
        stats.loc[f"P{int(p*100)}"] = (
            df["Particle_Size_mm"].quantile(p)
        )

    st.dataframe(
        stats.round(3),
        use_container_width=True
    )


# -------------------------------------------------------------------
# CLASSIFICATION
# -------------------------------------------------------------------
with classification_tab:
    st.subheader("🏷️ Automatic Coal Particle-Size Classification")

    st.info(
        f"""
        **Fine:** < {fine_limit:g} mm  
        **Small:** {fine_limit:g}–<{small_limit:g} mm  
        **Medium:** {small_limit:g}–<{medium_limit:g} mm  
        **Large:** {medium_limit:g}–<{large_limit:g} mm  
        **Coarse:** ≥ {large_limit:g} mm
        """
    )

    class_summary = (
        df.groupby("Particle_Size_Class", observed=False)
        .agg(
            Number_of_Particles=("Particle_Size_mm", "size"),
            Mean_Size_mm=("Particle_Size_mm", "mean"),
            Minimum_mm=("Particle_Size_mm", "min"),
            Maximum_mm=("Particle_Size_mm", "max")
        )
        .reindex(CLASS_LABELS)
    )

    class_summary["Percentage_%"] = (
        class_summary["Number_of_Particles"] / len(df) * 100
    )

    st.dataframe(
        class_summary.round(3),
        use_container_width=True
    )

    st.subheader("Classified particle records")
    st.dataframe(
        df.head(5000),
        use_container_width=True,
        hide_index=True
    )

    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(
        excel_buffer,
        engine="openpyxl"
    ) as writer:
        df.to_excel(
            writer,
            index=False,
            sheet_name="Classified_Data"
        )
        class_summary.round(3).to_excel(
            writer,
            sheet_name="Class_Summary"
        )

    st.download_button(
        "⬇️ Download Classified Excel",
        data=excel_buffer.getvalue(),
        file_name="coal_particle_size_classification.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        )
    )

    st.download_button(
        "⬇️ Download Classified CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="coal_particle_size_classification.csv",
        mime="text/csv"
    )


# -------------------------------------------------------------------
# COAL QUALITY
# -------------------------------------------------------------------
with quality_tab:
    st.subheader("🧪 Coal Quality Relationships")

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    quality_columns = [
        c for c in numeric_columns
        if c != "Particle_Size_mm"
    ]

    if quality_columns:
        selected = st.selectbox(
            "Select coal-quality variable",
            quality_columns
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(
            df["Particle_Size_mm"],
            df[selected],
            alpha=0.35,
            s=8
        )
        ax.set_xlabel("Particle size (mm)")
        ax.set_ylabel(selected)
        ax.set_title(
            "Particle Size vs " + selected
        )
        ax.grid(alpha=0.25)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    if len(numeric_columns) >= 2:
        st.subheader("Correlation Matrix")
        st.dataframe(
            df[numeric_columns].corr().round(3),
            use_container_width=True
        )


# -------------------------------------------------------------------
# PREDICTION
# -------------------------------------------------------------------
with prediction_tab:
    st.subheader("🔮 Particle-Size Class Prediction")

    particle_size_input = st.number_input(
        "Enter particle size (mm)",
        min_value=0.001,
        value=3.0,
        step=0.1
    )

    predicted = classify_particles(
        pd.Series([particle_size_input]),
        fine_limit,
        small_limit,
        medium_limit,
        large_limit
    ).iloc[0]

    st.metric(
        "Predicted Particle-Size Class",
        str(predicted)
    )

    st.info(
        "The current classifier is transparent and "
        "rule-based. The thresholds can be changed in "
        "the sidebar."
    )


# -------------------------------------------------------------------
# HOPPER ANIMATION
# -------------------------------------------------------------------
with hopper_tab:
    st.subheader("🎬 Live Coal Hopper Animation")
    st.caption(
        "Animated representation of coal particles entering "
        "a hopper, moving downward and leaving through the outlet."
    )

    controls_left, controls_mid, controls_right = st.columns(3)

    with controls_left:
        animation_particles = st.slider(
            "Particles visible in animation",
            min_value=100,
            max_value=2000,
            value=700,
            step=100
        )

    with controls_mid:
        animation_frames = st.slider(
            "Animation frames",
            min_value=20,
            max_value=150,
            value=70,
            step=10
        )

    with controls_right:
        speed = st.slider(
            "Animation speed",
            min_value=0.01,
            max_value=0.20,
            value=0.04,
            step=0.01
        )

    run_animation = st.button(
        "▶ Start Hopper Animation",
        type="primary",
        use_container_width=True
    )

    # Classification summary
    counts = (
        df["Particle_Size_Class"]
        .value_counts()
        .reindex(CLASS_LABELS, fill_value=0)
    )
    percentages = counts / len(df) * 100

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Fine", f"{percentages['Fine']:.1f}%")
    m2.metric("Small", f"{percentages['Small']:.1f}%")
    m3.metric("Medium", f"{percentages['Medium']:.1f}%")
    m4.metric("Large", f"{percentages['Large']:.1f}%")
    m5.metric("Coarse", f"{percentages['Coarse']:.1f}%")

    st.markdown("### Classification thresholds")
    threshold_table = pd.DataFrame({
        "Class": CLASS_LABELS,
        "Particle size": [
            f"< {fine_limit:g} mm",
            f"{fine_limit:g}–< {small_limit:g} mm",
            f"{small_limit:g}–< {medium_limit:g} mm",
            f"{medium_limit:g}–< {large_limit:g} mm",
            f"≥ {large_limit:g} mm"
        ]
    })
    st.dataframe(
        threshold_table,
        use_container_width=True,
        hide_index=True
    )

    animation_area = st.empty()
    status_area = st.empty()

    if run_animation:
        rng = np.random.default_rng()

        # Use a representative sample so the browser/server stays fast.
        sample_n = min(animation_particles, len(df))
        particles = df.sample(
            sample_n,
            random_state=int(rng.integers(0, 1_000_000))
        ).copy()

        sizes = particles["Particle_Size_mm"].to_numpy()
        classes = particles["Particle_Size_Class"].astype(str).to_numpy()

        # Normalize display sizes so large particles are visible but
        # do not dominate the plot.
        point_sizes = np.clip(
            18 + sizes * 5,
            18,
            90
        )

        class_color = {
            "Fine": "#1f77b4",
            "Small": "#2ca02c",
            "Medium": "#7f7f7f",
            "Large": "#ff7f0e",
            "Coarse": "#d62728"
        }

        # Initial particle positions inside the upper hopper.
        x = rng.uniform(-3.8, 3.8, sample_n)
        y = rng.uniform(1.0, 5.2, sample_n)

        # Keep initial particles inside the hopper walls.
        half_width = 4.0 * np.clip(
            (y + 0.2) / 5.4,
            0.15,
            1.0
        )
        x = np.clip(x, -half_width, half_width)

        # Larger particles fall slightly differently.
        fall_speed = (
            0.035
            + 0.025 * np.clip(sizes / 10, 0.2, 2.0)
        )

        for frame in range(animation_frames):
            y -= fall_speed * (1.0 + 0.20 * np.sin(frame / 5))

            # Reset particles that have reached the outlet.
            exited = y < -0.4
            if exited.any():
                y[exited] = rng.uniform(4.5, 5.3, exited.sum())
                x[exited] = rng.uniform(
                    -3.7,
                    3.7,
                    exited.sum()
                )

            # Constrain particles to the hopper shape.
            inside = y >= 0.2
            half_width = np.maximum(
                0.30,
                4.0 * ((y + 0.2) / 5.2)
            )
            x[inside] = np.clip(
                x[inside],
                -half_width[inside],
                half_width[inside]
            )

            fig, ax = plt.subplots(
                figsize=(12, 6.7)
            )

            # Hopper body.
            hopper = Polygon(
                [
                    (-4.2, 5.5),
                    (4.2, 5.5),
                    (0.75, 0.35),
                    (0.75, -0.35),
                    (-0.75, -0.35),
                    (-0.75, 0.35)
                ],
                closed=True,
                facecolor="#34383d",
                edgecolor="#111111",
                linewidth=3
            )
            ax.add_patch(hopper)

            # Inner hopper.
            inner = Polygon(
                [
                    (-3.9, 5.15),
                    (3.9, 5.15),
                    (0.55, 0.55),
                    (-0.55, 0.55)
                ],
                closed=True,
                facecolor="#20252a",
                edgecolor="#555555",
                linewidth=1.5
            )
            ax.add_patch(inner)

            # Top feed bar.
            ax.add_patch(
                Rectangle(
                    (-4.35, 5.35),
                    8.7,
                    0.35,
                    facecolor="#202124",
                    edgecolor="#111111",
                    linewidth=2
                )
            )

            # Outlet.
            ax.add_patch(
                Rectangle(
                    (-0.75, -0.55),
                    1.5,
                    0.35,
                    facecolor="#202124",
                    edgecolor="#111111",
                    linewidth=2
                )
            )

            # Particle colors.
            particle_colors = [
                class_color.get(c, "#777777")
                for c in classes
            ]

            ax.scatter(
                x,
                y,
                s=point_sizes,
                c=particle_colors,
                alpha=0.90,
                edgecolors="#111111",
                linewidths=0.35
            )

            # Direction arrow.
            ax.annotate(
                "",
                xy=(0, -1.35),
                xytext=(0, -0.55),
                arrowprops=dict(
                    arrowstyle="->",
                    linewidth=3
                )
            )

            ax.text(
                0,
                6.05,
                "COAL FEED",
                ha="center",
                va="center",
                fontsize=17,
                fontweight="bold"
            )

            ax.text(
                0,
                -1.65,
                "Fine Coal / Classified Coal Outlet",
                ha="center",
                va="center",
                fontsize=14
            )

            # Legend.
            legend_handles = []
            for label in CLASS_LABELS:
                handle = plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=class_color[label],
                    markeredgecolor="#111111",
                    markersize=9,
                    label=label
                )
                legend_handles.append(handle)

            ax.legend(
                handles=legend_handles,
                title="Particle class",
                loc="upper right",
                bbox_to_anchor=(1.20, 1.0)
            )

            ax.set_xlim(-5.2, 5.2)
            ax.set_ylim(-2.0, 6.5)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(
                "Live Coal Hopper Particle Flow",
                fontsize=19,
                fontweight="bold",
                pad=12
            )

            plt.tight_layout()

            animation_area.pyplot(
                fig,
                use_container_width=True
            )
            plt.close(fig)

            status_area.info(
                f"Animation frame {frame + 1}/{animation_frames} | "
                f"Showing {sample_n:,} particles | "
                f"Full dataset: {len(df):,} particles"
            )

            time.sleep(speed)

        status_area.success(
            "Hopper animation completed."
        )

    else:
        # Static preview before the user starts animation.
        rng = np.random.default_rng(10)
        preview_n = min(animation_particles, len(df))
        preview = df.sample(
            preview_n,
            random_state=10
        )

        class_color = {
            "Fine": "#1f77b4",
            "Small": "#2ca02c",
            "Medium": "#7f7f7f",
            "Large": "#ff7f0e",
            "Coarse": "#d62728"
        }

        fig, ax = plt.subplots(figsize=(12, 6.7))

        hopper = Polygon(
            [
                (-4.2, 5.5),
                (4.2, 5.5),
                (0.75, 0.35),
                (0.75, -0.35),
                (-0.75, -0.35),
                (-0.75, 0.35)
            ],
            closed=True,
            facecolor="#34383d",
            edgecolor="#111111",
            linewidth=3
        )
        ax.add_patch(hopper)

        inner = Polygon(
            [
                (-3.9, 5.15),
                (3.9, 5.15),
                (0.55, 0.55),
                (-0.55, 0.55)
            ],
            closed=True,
            facecolor="#20252a",
            edgecolor="#555555",
            linewidth=1.5
        )
        ax.add_patch(inner)

        ax.add_patch(
            Rectangle(
                (-4.35, 5.35),
                8.7,
                0.35,
                facecolor="#202124",
                edgecolor="#111111",
                linewidth=2
            )
        )

        sizes = preview["Particle_Size_mm"].to_numpy()
        y = rng.uniform(0.7, 5.0, preview_n)
        half_width = np.maximum(
            0.3,
            3.8 * ((y + 0.2) / 5.2)
        )
        x = rng.uniform(-1, 1, preview_n) * half_width

        colors = [
            class_color.get(str(c), "#777777")
            for c in preview["Particle_Size_Class"]
        ]

        ax.scatter(
            x,
            y,
            s=np.clip(18 + sizes * 5, 18, 90),
            c=colors,
            alpha=0.9,
            edgecolors="#111111",
            linewidths=0.35
        )

        ax.text(
            0, 6.05,
            "COAL FEED",
            ha="center",
            fontsize=17,
            fontweight="bold"
        )

        ax.text(
            0, -1.55,
            "Press ▶ Start Hopper Animation",
            ha="center",
            fontsize=14
        )

        ax.set_xlim(-5.2, 5.2)
        ax.set_ylim(-2.0, 6.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            "Fine vs Coarse Coal Flow Animation",
            fontsize=19,
            fontweight="bold"
        )

        plt.tight_layout()
        animation_area.pyplot(
            fig,
            use_container_width=True
        )
        plt.close(fig)

st.divider()

st.caption(
    "Research prototype | 150,000 synthetic coal particles | "
    "January–June 2026 | Working days Monday–Friday, 08:00–16:30"
)
'''

requirements = """streamlit>=1.40
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
openpyxl>=3.1
"""

(base / "app.py").write_text(app_code, encoding="utf-8")
(base / "requirements.txt").write_text(requirements, encoding="utf-8")

print("Created:", base / "app.py")
print("Created:", base / "requirements.txt")
print("app.py lines:", len(app_code.splitlines()))
