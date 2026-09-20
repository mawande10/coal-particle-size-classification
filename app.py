with animation_tab:
    st.subheader("🎥 Hopper Animation — Fine vs Coarse Coal Flow")

    hopper_width = 10
    hopper_height = 15

    def generate_positions(df):
        positions = []
        for _, row in df.iterrows():
            size_class = row["Particle_Size_Class"]
            y = np.random.uniform(0, hopper_height)
            if size_class == "Fine":
                x = np.random.normal(0, 1)
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
    ax.set_ylim(-2, hopper_height)

    # Draw hopper funnel walls
    ax.plot([-hopper_width/2, 0], [hopper_height, 0], color="black", linewidth=2)
    ax.plot([hopper_width/2, 0], [hopper_height, 0], color="black", linewidth=2)
    ax.fill_between([-hopper_width/2, hopper_width/2], hopper_height, 0, color="lightgray", alpha=0.2)

    # Collector bin
    ax.fill_between([-2, 2], -2, 0, color="darkgray", alpha=0.3)

    scat = ax.scatter([], [], s=[], alpha=0.7)

    def update(frame):
        xs, ys, sizes, colors = [], [], [], []
        for pos in positions:
            x, y, cls, size = pos
            y -= 0.1  # move downward
            funnel_ratio = max(0, (hopper_height - y) / hopper_height)
            if cls == "Fine":
                x *= (1 - 0.7 * funnel_ratio)
            elif cls == "Coarse":
                x *= (1 - 0.2 * funnel_ratio)
            else:
                x *= (1 - 0.5 * funnel_ratio)

            # Reset particle if it exits
            if y <= -2:
                y = hopper_height
                if cls == "Fine":
                    x = np.random.normal(0, 1)
                elif cls == "Coarse":
                    x = np.random.choice([-hopper_width/2, hopper_width/2]) + np.random.normal(0, 0.5)
                else:
                    x = np.random.uniform(-hopper_width/2, hopper_width/2)

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

    st.image("hopper_animation.gif", caption="Coal particles funnel down and discharge continuously.")


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
