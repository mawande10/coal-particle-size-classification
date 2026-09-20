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
