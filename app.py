import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Coal Particle Size Classification",
    page_icon="🪨",
    layout="wide"
)

st.title("🪨 Coal Particle Size Classification Dashboard")
st.caption(
    "Coal particle-size distribution, classification, "
    "coal-quality analysis and particle-size prediction"
)


# ============================================================
# CONSTANTS
# ============================================================

CLASS_LABELS = [
    "Fine",
    "Small",
    "Medium",
    "Large",
    "Coarse"
]


# ============================================================
# SAMPLE DATA GENERATOR
# ============================================================

def make_sample_data(n=1000, seed=42):

    rng = np.random.default_rng(seed)

    # Generate realistic positive particle sizes
    particle_size = np.clip(
        rng.lognormal(
            mean=np.log(3.2),
            sigma=0.8,
            size=n
        ),
        0.05,
        30
    )

    mass = np.clip(
        rng.gamma(
            shape=2.5,
            scale=12,
            size=n
        ),
        0.5,
        100
    )

    moisture = np.clip(
        7.5
        - 0.06 * particle_size
        + rng.normal(0, 1, n),
        2,
        15
    )

    ash = np.clip(
        15
        + 0.7 * np.log1p(particle_size)
        + rng.normal(0, 3, n),
        5,
        35
    )

    density = np.clip(
        1.35
        + 0.025 * np.log1p(particle_size)
        + rng.normal(0, 0.06, n),
        1.15,
        1.90
    )

    calorific_value = np.clip(
        27
        - 0.25 * ash
        - 0.25 * moisture
        + rng.normal(0, 1.2, n),
        15,
        32
    )

    return pd.DataFrame({
        "Sample_ID": [
            f"COAL-{i:04d}"
            for i in range(1, n + 1)
        ],

        "Particle_Size_mm":
            np.round(particle_size, 3),

        "Mass_g":
            np.round(mass, 2),

        "Moisture_%":
            np.round(moisture, 2),

        "Density_g_cm3":
            np.round(density, 3),

        "Ash_%":
            np.round(ash, 2),

        "Calorific_Value_MJ_kg":
            np.round(calorific_value, 2)
    })


# ============================================================
# COLUMN STANDARDISATION
# ============================================================

def standardise_columns(df):

    df = df.copy()

    df.columns = [
        str(column)
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        for column in df.columns
    ]

    aliases = {

        "Particle_Size":
            "Particle_Size_mm",

        "Size_mm":
            "Particle_Size_mm",

        "ParticleSize_mm":
            "Particle_Size_mm",

        "ParticleSize":
            "Particle_Size_mm",

        "Mass":
            "Mass_g",

        "Moisture":
            "Moisture_%",

        "Ash":
            "Ash_%",

        "Density":
            "Density_g_cm3",

        "Calorific_Value":
            "Calorific_Value_MJ_kg",

        "CV_MJ_kg":
            "Calorific_Value_MJ_kg"
    }

    for old_name, new_name in aliases.items():

        if (
            old_name in df.columns
            and new_name not in df.columns
        ):

            df.rename(
                columns={
                    old_name: new_name
                },
                inplace=True
            )

    return df


# ============================================================
# PARTICLE CLASSIFICATION FUNCTION
# ============================================================

def classify_particles(
    particle_sizes,
    fine_limit,
    small_limit,
    medium_limit,
    large_limit
):

    bins = [
        -np.inf,
        fine_limit,
        small_limit,
        medium_limit,
        large_limit,
        np.inf
    ]

    return pd.cut(
        particle_sizes,
        bins=bins,
        labels=CLASS_LABELS,
        right=False,
        include_lowest=True
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Dashboard Controls")


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.sidebar.file_uploader(
    "Upload coal particle-size data",
    type=[
        "xlsx",
        "xls",
        "csv"
    ]
)


# ============================================================
# READ DATA
# ============================================================

if uploaded_file is None:

    df = make_sample_data()

    st.sidebar.info(
        "No file uploaded. "
        "The dashboard is using the built-in "
        "sample coal dataset."
    )

else:

    try:

        if uploaded_file.name.lower().endswith(".csv"):

            df = pd.read_csv(
                uploaded_file
            )

        else:

            df = pd.read_excel(
                uploaded_file
            )

        df = standardise_columns(df)

        st.sidebar.success(
            f"Loaded: {uploaded_file.name}"
        )

    except Exception as error:

        st.error(
            f"Could not read the uploaded file: {error}"
        )

        st.stop()


# ============================================================
# CHECK PARTICLE SIZE COLUMN
# ============================================================

if "Particle_Size_mm" not in df.columns:

    st.error(
        """
        The uploaded dataset must contain a particle-size column.

        Accepted names:

        • Particle_Size_mm
        • Particle_Size
        • Size_mm
        • ParticleSize_mm
        • ParticleSize
        """
    )

    st.stop()


# ============================================================
# CLEAN PARTICLE SIZE
# ============================================================

df["Particle_Size_mm"] = pd.to_numeric(
    df["Particle_Size_mm"],
    errors="coerce"
)

df = df.dropna(
    subset=["Particle_Size_mm"]
).copy()


if df.empty:

    st.error(
        "No valid particle-size values were found."
    )

    st.stop()


# ============================================================
# CLASSIFICATION THRESHOLDS
# ============================================================

st.sidebar.subheader(
    "Particle-size classification"
)

st.sidebar.write(
    "Adjust the particle-size boundaries "
    "below to match your research methodology."
)


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


# ============================================================
# APPLY CLASSIFICATION
# ============================================================

df["Particle_Size_Class"] = classify_particles(
    df["Particle_Size_mm"],
    fine_limit,
    small_limit,
    medium_limit,
    large_limit
)


# ============================================================
# DASHBOARD TABS
# ============================================================

overview_tab, distribution_tab, classification_tab, quality_tab, prediction_tab = st.tabs(
    [
        "📊 Overview",
        "📈 Size Distribution",
        "🏷️ Classification",
        "🧪 Coal Quality",
        "🔮 Prediction"
    ]
)


# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with overview_tab:

    st.subheader(
        "Coal Particle-Size Overview"
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Number of Samples",
        f"{len(df):,}"
    )

    col2.metric(
        "Mean Size",
        f"{df['Particle_Size_mm'].mean():.2f} mm"
    )

    col3.metric(
        "Median Size",
        f"{df['Particle_Size_mm'].median():.2f} mm"
    )

    col4.metric(
        "Minimum Size",
        f"{df['Particle_Size_mm'].min():.2f} mm"
    )

    col5.metric(
        "Maximum Size",
        f"{df['Particle_Size_mm'].max():.2f} mm"
    )


    st.divider()


    # Class counts

    class_counts = (
        df["Particle_Size_Class"]
        .value_counts()
        .reindex(
            CLASS_LABELS,
            fill_value=0
        )
    )


    class_percentages = (
        class_counts
        / len(df)
        * 100
    )


    left_column, right_column = st.columns(2)


    # ----------------------------
    # BAR CHART
    # ----------------------------

    with left_column:

        st.subheader(
            "Particle-size classification"
        )

        fig, ax = plt.subplots(
            figsize=(8, 4.5)
        )

        ax.bar(
            CLASS_LABELS,
            class_counts.values
        )

        ax.set_xlabel(
            "Particle-size class"
        )

        ax.set_ylabel(
            "Number of particles"
        )

        ax.set_title(
            "Coal Particle-size Classification"
        )

        ax.tick_params(
            axis="x",
            rotation=20
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


    # ----------------------------
    # SUMMARY TABLE
    # ----------------------------

    with right_column:

        st.subheader(
            "Classification summary"
        )

        summary_table = pd.DataFrame({

            "Class":
                CLASS_LABELS,

            "Number_of_Particles":
                class_counts.values,

            "Percentage_%":
                np.round(
                    class_percentages.values,
                    2
                )
        })

        st.dataframe(
            summary_table,
            use_container_width=True,
            hide_index=True
        )


    # ----------------------------
    # DATA PREVIEW
    # ----------------------------

    st.subheader(
        "Dataset preview"
    )

    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 2 — PARTICLE SIZE DISTRIBUTION
# ============================================================

with distribution_tab:

    st.subheader(
        "📈 Particle-Size Distribution"
    )


    # Histogram

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.hist(
        df["Particle_Size_mm"],
        bins=30
    )

    ax.set_xlabel(
        "Particle size (mm)"
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.set_title(
        "Coal Particle-Size Distribution"
    )

    ax.grid(
        alpha=0.25
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    # Statistics

    st.subheader(
        "Particle-size statistics"
    )


    statistics = (
        df["Particle_Size_mm"]
        .describe()
        .to_frame(
            "Particle_Size_mm"
        )
    )


    statistics.loc["P10"] = (
        df["Particle_Size_mm"]
        .quantile(0.10)
    )

    statistics.loc["P25"] = (
        df["Particle_Size_mm"]
        .quantile(0.25)
    )

    statistics.loc["P50"] = (
        df["Particle_Size_mm"]
        .quantile(0.50)
    )

    statistics.loc["P75"] = (
        df["Particle_Size_mm"]
        .quantile(0.75)
    )

    statistics.loc["P90"] = (
        df["Particle_Size_mm"]
        .quantile(0.90)
    )


    st.dataframe(
        statistics.round(3),
        use_container_width=True
    )


# ============================================================
# TAB 3 — CLASSIFICATION
# ============================================================

with classification_tab:

    st.subheader(
        "🏷️ Automatic Coal Particle-Size Classification"
    )


    st.info(
        f"""
        Current classification:

        Fine < {fine_limit:g} mm

        Small {fine_limit:g}–<{small_limit:g} mm

        Medium {small_limit:g}–<{medium_limit:g} mm

        Large {medium_limit:g}–<{large_limit:g} mm

        Coarse ≥ {large_limit:g} mm
        """
    )


    # Classification summary

    class_summary = (
        df.groupby(
            "Particle_Size_Class",
            observed=False
        )
        .agg(
            Number_of_Particles=(
                "Particle_Size_mm",
                "size"
            ),

            Mean_Size_mm=(
                "Particle_Size_mm",
                "mean"
            ),

            Minimum_mm=(
                "Particle_Size_mm",
                "min"
            ),

            Maximum_mm=(
                "Particle_Size_mm",
                "max"
            )
        )
        .reindex(CLASS_LABELS)
    )


    class_summary["Percentage_%"] = (
        class_summary["Number_of_Particles"]
        / len(df)
        * 100
    )


    st.dataframe(
        class_summary.round(3),
        use_container_width=True
    )


    # Full classified dataset

    st.subheader(
        "Classified particle records"
    )


    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # EXCEL DOWNLOAD
    # ========================================================

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
        label="⬇️ Download Classified Excel",
        data=excel_buffer.getvalue(),
        file_name=(
            "coal_particle_size_classification.xlsx"
        ),
        mime=(
            "application/vnd.openxmlformats-"
            "officedocument.spreadsheetml.sheet"
        )
    )


    # ========================================================
    # CSV DOWNLOAD
    # ========================================================

    csv_data = df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="⬇️ Download Classified CSV",
        data=csv_data,
        file_name=(
            "coal_particle_size_classification.csv"
        ),
        mime="text/csv"
    )


# ============================================================
# TAB 4 — COAL QUALITY
# ============================================================

with quality_tab:

    st.subheader(
        "🧪 Coal Quality Relationships"
    )


    numeric_columns = (
        df.select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )


    quality_columns = [
        column
        for column in numeric_columns
        if column != "Particle_Size_mm"
    ]


    if quality_columns:

        selected_variable = st.selectbox(
            "Select coal-quality variable",
            quality_columns
        )


        fig, ax = plt.subplots(
            figsize=(9, 5)
        )


        ax.scatter(
            df["Particle_Size_mm"],
            df[selected_variable],
            alpha=0.65
        )


        ax.set_xlabel(
            "Particle size (mm)"
        )

        ax.set_ylabel(
            selected_variable
        )

        ax.set_title(
            "Particle Size vs "
            + selected_variable
        )

        ax.grid(
            alpha=0.25
        )


        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


    # Correlation matrix

    if len(numeric_columns) >= 2:

        st.subheader(
            "Correlation Matrix"
        )


        correlation = (
            df[numeric_columns]
            .corr()
        )


        st.dataframe(
            correlation.round(3),
            use_container_width=True
        )


    else:

        st.info(
            """
            Add additional numerical variables such as:

            • Ash
            • Moisture
            • Density
            • Calorific value
            • Mass
            """
        )


# ============================================================
# TAB 5 — PREDICTION
# ============================================================

with prediction_tab:

    st.subheader(
        "🔮 Particle-Size Class Prediction"
    )


    particle_size_input = st.number_input(
        "Enter particle size (mm)",
        min_value=0.001,
        value=3.0,
        step=0.1
    )


    predicted_class = pd.cut(
        [particle_size_input],
        bins=[
            -np.inf,
            fine_limit,
            small_limit,
            medium_limit,
            large_limit,
            np.inf
        ],
        labels=CLASS_LABELS,
        right=False,
        include_lowest=True
    )[0]


    st.metric(
        "Predicted Particle-Size Class",
        str(predicted_class)
    )


    st.info(
        """
        This dashboard currently uses a transparent,
        rule-based particle-size classifier.

        The thresholds can be changed in the sidebar.

        A future version can use machine learning such as
        Random Forest, XGBoost or image-based particle
        classification.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Coal Particle Size Classification Dashboard | "
    "Streamlit Research Prototype"
)
