import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(
    page_title="Uzawa–Lucas Model Simulator",
    layout="wide"
)

st.title("📈 Uzawa–Lucas Model Simulator")
st.markdown(
    """
This application lets you explore the **Uzawa–Lucas model** under:

- a **competitive equilibrium** (households ignore the human-capital externality),
- a **social planner** (who internalizes the external effect of average human capital).

Use the **parameter panel on the left** to change technology, preferences and depreciation,  
and see how the growth rates and time paths react.
"""
)

# -------------------------------------------------------
# SIDEBAR — PARAMETER PANEL
# -------------------------------------------------------
st.sidebar.header("Model parameters")

beta = st.sidebar.slider("β — capital share", 0.05, 0.9, 0.30, 0.01)
gamma = st.sidebar.slider("γ — strength of human-capital externality", 0.00, 0.6, 0.20, 0.01)
delta = st.sidebar.slider("δ — human-capital technology", 0.01, 0.20, 0.08, 0.01)
rho_minus_lambda = st.sidebar.slider("ρ − λ — effective discount rate", 0.00, 0.20, 0.04, 0.01)
lambda_param = st.sidebar.slider("λ — physical-capital depreciation", 0.00, 0.10, 0.01, 0.005)
theta = st.sidebar.slider("θ — inverse IES (risk aversion)", 0.5, 5.0, 2.0, 0.1)

T = st.sidebar.number_input("Time horizon T", min_value=50, max_value=1000, value=200, step=10)
dt = st.sidebar.number_input("Time step dt", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
N = int(T / dt)


# -------------------------------------------------------
# GROWTH RATE FUNCTIONS: ν (equilibrium) and ν* (planner)
# -------------------------------------------------------
def nu_equilibrium(theta_val: float) -> float:
    """
    Competitive equilibrium growth rate of human capital:
    ν = (1−β)(δ − (ρ−λ)) / [θ(1−β+γ) − γ], clipped to [0, δ].
    """
    num = (1.0 - beta) * (delta - rho_minus_lambda)
    den = theta_val * (1.0 - beta + gamma) - gamma
    if den <= 0:
        return 0.0
    nu_val = num / den
    return float(np.clip(nu_val, 0.0, delta))


def nu_planner(theta_val: float) -> float:
    """
    Planner's optimal growth rate of human capital:
    ν* = [δ − (1−β)/(1−β+γ) (ρ−λ)] / θ, clipped to [0, δ].
    """
    base = delta - (1.0 - beta) / (1.0 - beta + gamma) * rho_minus_lambda
    nu_star_val = base / theta_val
    return float(np.clip(nu_star_val, 0.0, delta))


# -------------------------------------------------------
# DYNAMICS SIMULATION (equilibrium or planner)
# -------------------------------------------------------
def simulate(is_planner: bool = False):
    """
    Simulate time paths for (k, h, u, c, y)
    using a simple adjustment rule where u(t)
    gradually moves toward its steady-state value
    implied by ν or ν*.
    """
    k = np.zeros(N)
    h = np.zeros(N)
    u = np.zeros(N)
    c = np.zeros(N)
    y = np.zeros(N)

    # Initial conditions
    k[0] = 1.0
    h[0] = 1.0
    u[0] = 0.4

    # Choose growth rate ν according to regime
    if is_planner:
        nu_val = nu_planner(theta)
    else:
        nu_val = nu_equilibrium(theta)

    # Steady-state time allocation u*
    if delta > 0:
        u_star = 1.0 - nu_val / delta
    else:
        u_star = 0.0
    u_star = float(np.clip(u_star, 0.0, 1.0))

    for t in range(N - 1):
        # Output
        y[t] = k[t] ** beta * (u[t] * h[t]) ** (1.0 - beta) * h[t] ** gamma

        # Simple consumption rule: constant share of output
        c[t] = 0.3 * y[t]

        # Laws of motion
        k[t + 1] = k[t] + dt * (y[t] - c[t] - lambda_param * k[t])
        h[t + 1] = h[t] + dt * (delta * (1.0 - u[t]) * h[t])

        # Gradual adjustment of u(t) toward u*
        u[t + 1] = u[t] + dt * 0.1 * (u_star - u[t])

    # Final period output and consumption
    y[-1] = k[-1] ** beta * (u[-1] * h[-1]) ** (1.0 - beta) * h[-1] ** gamma
    c[-1] = 0.3 * y[-1]

    return k, h, u, c, y


# -------------------------------------------------------
# 1) ANALYTICAL PLOTS: κ/ν and ν, ν*
# -------------------------------------------------------
st.header("1. Analytical growth relationships")

col1, col2 = st.columns(2)

# --- FIGURE 1: κ/ν as a function of γ ---
with col1:
    st.subheader("1a. Balanced-growth relation κ/ν")
    gamma_grid = np.linspace(0.0, 0.4, 300)
    kappa_over_nu = (1.0 - beta + gamma_grid) / (1.0 - beta)

    fig1, ax1 = plt.subplots(figsize=(5, 4))
    ax1.plot(gamma_grid, kappa_over_nu, linewidth=2)
    ax1.axhline(1.0, linestyle="--", color="gray")
    ax1.set_xlabel(r"$\gamma$")
    ax1.set_ylabel(r"$\kappa / \nu$")
    ax1.set_title(r"$\kappa / \nu = \dfrac{1-\beta+\gamma}{1-\beta}$")
    st.pyplot(fig1)

    st.markdown(
        """
**Interpretation**

- When **γ = 0**, there is no external effect from average human capital and  
  we obtain **κ = ν**, so physical and human capital (and consumption) grow at the same rate.
- When **γ > 0**, external learning spillovers make **physical capital grow faster** than human capital.
        """
    )

# --- FIGURE 2: ν* vs ν as functions of θ ---
with col2:
    st.subheader("1b. Optimal vs equilibrium human-capital growth")

    theta_grid = np.linspace(0.5, 4.0, 400)
    nu_star_grid = np.array([nu_planner(th) for th in theta_grid])
    nu_grid = np.array([nu_equilibrium(th) for th in theta_grid])

    # knife-edge θ where ν* = ν = δ
    if delta > 0:
        theta_star = 1.0 - (1.0 - beta) / (1.0 - beta + gamma) * (rho_minus_lambda / delta)
    else:
        theta_star = np.nan

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.plot(theta_grid, nu_star_grid, label=r"$\nu^*$ (planner)", linewidth=2)
    ax2.plot(theta_grid, nu_grid, linestyle="--", label=r"$\nu$ (equilibrium)", linewidth=2)
    ax2.axhline(delta, color="gray", linestyle=":", label=r"upper bound $\nu = \delta$")
    if np.isfinite(theta_star):
        ax2.axvline(theta_star, linestyle=":", color="black",
                    label=rf"$\theta^* \approx {theta_star:.2f}$")
    ax2.set_xlabel(r"$\theta$")
    ax2.set_ylabel("growth rate of human capital")
    ax2.set_title(r"Optimal vs equilibrium $g_h$ as functions of $\theta$")
    ax2.legend()
    st.pyplot(fig2)

    st.markdown(
        f"""
**Interpretation**

- The solid line shows the **planner’s optimal growth rate** of human capital, ν\*.
- The dashed line shows the **competitive equilibrium growth rate**, ν.
- For **γ > 0**, we typically have **ν < ν\***, meaning the equilibrium  
  **underinvests in education** relative to the social optimum.
- Both curves are capped at **δ = {delta:.3f}**, the **maximum feasible growth rate** of human capital.
- The vertical line at **θ\* ≈ {theta_star:.2f}** (when it exists) marks the knife-edge case 
  where **ν = ν\* = δ**.
        """
    )


# -------------------------------------------------------
# RUN SIMULATIONS (equilibrium and planner)
# -------------------------------------------------------
k_e, h_e, u_e, c_e, y_e = simulate(is_planner=False)
k_p, h_p, u_p, c_p, y_p = simulate(is_planner=True)
time = np.linspace(0.0, T, N)


# -------------------------------------------------------
# 2) TIME PATHS: k(t), c(t), y(t) — equilibrium
# -------------------------------------------------------
st.header("2. Time paths of capital, consumption, and output (equilibrium)")

fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.plot(time, k_e, label="k_e(t)")
ax3.plot(time, c_e, label="c_e(t)")
ax3.plot(time, y_e, label="y_e(t)")
ax3.set_xlabel("time")
ax3.set_title("Equilibrium time paths of k(t), c(t), and y(t)")
ax3.legend()
st.pyplot(fig3)

st.markdown(
    """
**Interpretation**

- These paths show how **physical capital**, **consumption**, and **output** evolve  
  when households choose time allocation competitively.
- Increasing **δ** raises the speed of human-capital accumulation and thus output growth.
- Increasing **γ** amplifies the contribution of average human capital to output,  
  strengthening the growth of **y(t)** and **k(t)**.
"""
)


# -------------------------------------------------------
# 3) TIME PATHS: u(t) and h(t) — equilibrium
# -------------------------------------------------------
st.header("3. Human capital and time allocation (equilibrium)")

fig4, ax4_left = plt.subplots(figsize=(8, 4))
ax4_left.plot(time, u_e, color="blue", label="u_e(t)")
ax4_left.set_xlabel("time")
ax4_left.set_ylabel("u_e(t)", color="blue")
ax4_right = ax4_left.twinx()
ax4_right.plot(time, h_e, color="orange", linestyle="--", label="h_e(t)")
ax4_right.set_ylabel("h_e(t)", color="orange")
ax4_left.set_title("Equilibrium time allocation and human capital")
# Combined legend
lines1, labels1 = ax4_left.get_legend_handles_labels()
lines2, labels2 = ax4_right.get_legend_handles_labels()
ax4_left.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
st.pyplot(fig4)

st.markdown(
    """
**Interpretation**

- The path **u_e(t)** shows how much time is spent in **production** rather than education.
- When **γ > 0**, households ignore the external payoff from human capital,  
  so they tend to choose **too high u(t)** (too much work, too little schooling).
- As a result, **h_e(t)** grows more slowly than under the planner solution.
"""
)


# -------------------------------------------------------
# 4) LOG-PATH COMPARISON: equilibrium vs planner
# -------------------------------------------------------
st.header("4. Equilibrium vs planner (log paths of k and h)")

# Avoid log of non-positive values
k_e_pos = np.where(k_e > 0, k_e, np.nan)
h_e_pos = np.where(h_e > 0, h_e, np.nan)
k_p_pos = np.where(k_p > 0, k_p, np.nan)
h_p_pos = np.where(h_p > 0, h_p, np.nan)

fig5, ax5 = plt.subplots(figsize=(8, 4))
ax5.plot(time, np.log(k_e_pos), label="log k_e(t) — equilibrium")
ax5.plot(time, np.log(h_e_pos), label="log h_e(t) — equilibrium")
ax5.plot(time, np.log(k_p_pos), linestyle="--", label="log k_p(t) — planner")
ax5.plot(time, np.log(h_p_pos), linestyle="--", label="log h_p(t) — planner")
ax5.set_xlabel("time")
ax5.set_ylabel("log levels")
ax5.set_title("Equilibrium vs planner: log paths of k(t) and h(t)")
ax5.legend()
st.pyplot(fig5)

st.markdown(
    """
**Interpretation**

- The **planner** internalizes the externality from average human capital (γ),  
  so she allocates **more time to education** than competitive households.
- As a result, both **log k_p(t)** and **log h_p(t)** are steeper than  
  their equilibrium counterparts, reflecting **higher long-run growth rates**.
- This illustrates quantitatively that the decentralized equilibrium **underinvests  
  in human capital** whenever **γ > 0**.
"""
)

st.success("Simulation and plots generated successfully. You can now experiment with the parameters on the left.")