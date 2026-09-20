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
