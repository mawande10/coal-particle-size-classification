# 🪨 Coal Particle Size Classification Dashboard

A Streamlit dashboard for analysing and classifying coal particle sizes.

The application is designed for coal-processing, laboratory particle-size analysis, DEM studies, material handling and research applications.

---

## Dashboard Features

The dashboard provides:

- Excel data upload
- CSV data upload
- Built-in sample coal dataset
- Automatic particle-size classification
- Editable particle-size thresholds
- Particle-size distribution histogram
- P10, P25, P50, P75 and P90 statistics
- Fine / Small / Medium / Large / Coarse classification
- Classification percentage
- Coal-quality analysis
- Particle size versus ash
- Particle size versus moisture
- Particle size versus density
- Particle size versus calorific value
- Correlation matrix
- Individual particle-size prediction
- Classified Excel download
- Classified CSV download

---

# Default Particle-Size Classification

The dashboard starts with the following operational classification:

| Particle Size | Classification |
|---|---|
| < 0.5 mm | Fine |
| 0.5 – <2 mm | Small |
| 2 – <5 mm | Medium |
| 5 – <10 mm | Large |
| >= 10 mm | Coarse |

These thresholds are configurable from the Streamlit sidebar.

For a research publication, replace the default thresholds with the particle-size boundaries specified by the relevant laboratory, coal-processing, screening or DEM methodology.

---

# Required Dataset

Your Excel or CSV file should contain a particle-size column.

The application accepts the following names:

```text
Particle_Size_mm
Particle_Size
Size_mm
ParticleSize_mm
ParticleSize
