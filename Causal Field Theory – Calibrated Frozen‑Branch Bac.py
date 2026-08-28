#!/usr/bin/env python3
"""
Causal Field Theory – Calibrated Frozen‑Branch Background Solution
==================================================================
This module integrates the corrected cosmological background equations
of Causal Field Theory using the physically calibrated frozen‑branch
parameters:

    β   = 0.002081       (potential scale)
    φ_i = -0.973092       (initial causal field)
    v_i =  0.360392       (initial field velocity)

The exact field equations in natural units (8πG/3 = 1) are:

    H² = e^{-4φ} ρ_m + ½(dφ/dt)² + V(φ)
    d²φ/dt² + 3H dφ/dt + V'(φ) = e^{-4φ} ρ_m
    dρ_m/dt + 3H ρ_m = 3 dφ/dt ρ_m

The code performs:
  · forward integration from a_i = 1e-3 to a = 1
  · verification of the frozen branch (w₀ ≈ –1)
  · luminosity distance and χ² comparison with the Pantheon+
    sample of 277 Hubble‑flow type Ia supernovae
  · CSV export of all numerical data
  · publication‑ready six‑panel figure with logarithmic redshift axes

Note: A separate validation suite (`validation_tests.py`) is provided
      for robustness checks and can be run independently.

Author : Mahmoud F. Abdel‑Sattar
Date   : 2026
"""

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, trapezoid
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# 1.  CALIBRATED FROZEN‑BRANCH PARAMETERS
# =============================================================================
BETA   = 0.002081          # Potential scale
PHI_I  = -0.973092          # Initial causal field
V_I    = 0.360392           # Initial field velocity

# =============================================================================
# 2.  FIXED QUANTITIES (natural units with 8πG/3 = 1)
# =============================================================================
OMEGA_M0 = 0.3              # Target present‑day matter density parameter
A_INIT   = 1e-3             # Initial scale factor
RHO_CRIT = 1.0              # Critical density in natural units
RHO_M0   = OMEGA_M0 * RHO_CRIT
RHO_M_I  = RHO_M0 / A_INIT**3   # Initial matter density

# =============================================================================
# 3.  POTENTIAL AND DERIVATIVE
# =============================================================================
def potential(phi):
    """Evaluate the causal potential V(φ)."""
    return BETA * (np.exp(2.0 * phi) - 1.0 - 2.0 * phi)


def potential_derivative(phi):
    """Evaluate the derivative dV/dφ."""
    return 2.0 * BETA * (np.exp(2.0 * phi) - 1.0)


# =============================================================================
# 4.  CORRECTED BACKGROUND EQUATIONS OF MOTION
# =============================================================================
def equations_of_motion(t, y):
    """
    Compute the right‑hand side of the autonomous system.

    Parameters
    ----------
    t : float
        Cosmic time.
    y : array_like
        State vector [a, φ, dφ/dt, ρ_m].

    Returns
    -------
    list
        Derivatives [da/dt, dφ/dt, d²φ/dt², dρ_m/dt].
    """
    a, phi, v, rho_m = y
    exp_4phi = np.exp(-4.0 * phi)

    # Friedmann constraint
    rho_total = exp_4phi * rho_m + 0.5 * v * v + potential(phi)
    if rho_total < 0.0:
        rho_total = 0.0
    H = np.sqrt(rho_total)

    da_dt   = a * H
    dphi_dt = v
    dv_dt   = -3.0 * H * v - potential_derivative(phi) + exp_4phi * rho_m
    drho_dt = -3.0 * H * rho_m + 3.0 * v * rho_m

    return [da_dt, dphi_dt, dv_dt, drho_dt]


# =============================================================================
# 5.  INTEGRATION ROUTINE
# =============================================================================
def integrate_background():
    """
    Integrate the background equations from a_init to a = 1.

    Returns
    -------
    dict
        Dictionary containing the numerical solution and derived quantities.
    """
    y0 = [A_INIT, PHI_I, V_I, RHO_M_I]

    def stop_at_present(t, y, *args):
        return y[0] - 1.0
    stop_at_present.terminal  = True
    stop_at_present.direction = 0

    sol = solve_ivp(equations_of_motion, [0.0, 30.0], y0,
                    method="RK45", events=stop_at_present,
                    max_step=0.01, rtol=1e-9, atol=1e-12)

    if not sol.success or abs(sol.y[0, -1] - 1.0) > 0.01:
        raise RuntimeError("Integration failed to reach a = 1.")

    a_arr     = sol.y[0]
    phi_arr   = sol.y[1]
    v_arr     = sol.y[2]
    rho_m_arr = sol.y[3]
    t_arr     = sol.t
    z_arr     = 1.0 / a_arr - 1.0

    exp_4phi_arr = np.exp(-4.0 * phi_arr)
    H_arr = np.sqrt(np.maximum(0.0,
                               exp_4phi_arr * rho_m_arr
                               + 0.5 * v_arr**2
                               + potential(phi_arr)))
    rho_phi_arr = 0.5 * v_arr**2 + potential(phi_arr)

    idx = -1
    H_today   = H_arr[idx]
    Omega_phi = rho_phi_arr[idx] / (H_today**2)
    w0 = ((0.5 * v_arr[idx]**2 - potential(phi_arr[idx]))
          / max(1e-12, 0.5 * v_arr[idx]**2 + potential(phi_arr[idx])))

    return {
        "a": a_arr, "phi": phi_arr, "v": v_arr, "rho_m": rho_m_arr,
        "t": t_arr, "z": z_arr, "H": H_arr, "rho_phi": rho_phi_arr,
        "H_today": H_today, "Omega_phi": Omega_phi, "w0": w0
    }


# =============================================================================
# 6.  LUMINOSITY DISTANCE & PANTHEON+ LIKELIHOOD
# =============================================================================
def compute_pantheon_likelihood(solution):
    """
    Compute χ² against the Pantheon+ Hubble‑flow supernova sample.

    Parameters
    ----------
    solution : dict
        Output of `integrate_background()`.

    Returns
    -------
    tuple
        (χ², degrees of freedom, residuals, z_sne, μ_obs, σ, μ_th, N_sne)
    """
    z_arr = solution["z"]
    H_arr = solution["H"]
    H_interp = interp1d(z_arr, H_arr, kind="linear",
                        bounds_error=False, fill_value="extrapolate")

    def luminosity_distance(z):
        if z <= 0.0:
            return 0.0
        z_grid = np.linspace(0.0, z, 500)
        return (1.0 + z) * trapezoid(1.0 / H_interp(z_grid), z_grid)

    z_fine = np.linspace(0.0, np.max(z_arr), 200)
    D_L_fine = np.array([luminosity_distance(zi) for zi in z_fine])
    D_L_interp = interp1d(z_fine, D_L_fine, kind="cubic")

    url = ("https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/"
           "main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat")
    data = pd.read_csv(url, sep=r"\s+", comment="#")
    data = data[data["USED_IN_SH0ES_HF"] == 1].copy()
    z_sne   = data["zHD"].values
    mu_obs  = data["MU_SH0ES"].values
    sigma   = data["MU_SH0ES_ERR_DIAG"].values
    N_sne   = len(z_sne)

    mu_th0 = 5.0 * np.log10(D_L_interp(z_sne))
    M_best = np.sum((mu_obs - mu_th0) / sigma**2) / np.sum(1.0 / sigma**2)
    mu_th  = mu_th0 + M_best

    chi2 = np.sum(((mu_obs - mu_th) / sigma)**2)
    dof  = N_sne - 1
    residuals = mu_obs - mu_th

    return chi2, dof, residuals, z_sne, mu_obs, sigma, mu_th, N_sne


# =============================================================================
# 7.  MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Causal Field Theory – Calibrated Frozen‑Branch Solution")
    print("=" * 60)
    print(f"β    = {BETA:.6f}")
    print(f"φ_i  = {PHI_I:.6f}")
    print(f"v_i  = {V_I:.6f}")
    print()

    solution = integrate_background()

    print("Verification numbers:")
    print(f"  H₀       = {solution['H_today']:.4f}")
    print(f"  Ω_φ      = {solution['Omega_phi']:.4f}")
    print(f"  w₀       = {solution['w0']:.4f}")
    if solution["w0"] < -0.9:
        print("  ✓ Frozen branch confirmed.")
    else:
        print("  ⚠ Warning: w₀ is not close to –1.")

    # Pantheon+ likelihood
    try:
        chi2_sn, dof, residuals_sn, z_sne, mu_obs, sigma, mu_th, N_sne = \
            compute_pantheon_likelihood(solution)
        print(f"\nPantheon+ χ² = {chi2_sn:.2f}  (dof = {dof})")
        pantheon_available = True
    except Exception:
        print("\nPantheon+ comparison skipped (data unavailable).")
        pantheon_available = False

    # ------------------------------------------------------------------
    # CSV EXPORT
    # ------------------------------------------------------------------
    z_arr = solution["z"]
    a_arr = solution["a"]
    H_arr = solution["H"]
    phi_arr = solution["phi"]
    v_arr = solution["v"]
    rho_m_arr = solution["rho_m"]
    rho_phi_arr = solution["rho_phi"]

    Omega_m = np.exp(-4.0 * phi_arr) * rho_m_arr / (H_arr**2)
    Omega_phi = rho_phi_arr / (H_arr**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        w_phi = ((0.5 * v_arr**2 - potential(phi_arr))
                 / np.maximum(1e-12, rho_phi_arr))
        w_phi = np.clip(w_phi, -2.0, 2.0)

    pd.DataFrame({
        "z": z_arr, "a": a_arr, "H": H_arr,
        "Omega_m": Omega_m, "Omega_phi": Omega_phi, "w_phi": w_phi,
        "phi": phi_arr, "rho_m": rho_m_arr, "rho_phi": rho_phi_arr
    }).to_csv("cosmology_background.csv", index=False)

    if pantheon_available:
        pd.DataFrame({
            "z": z_sne, "mu_obs": mu_obs, "sigma": sigma,
            "mu_CFT": mu_th, "residual_CFT": residuals_sn
        }).to_csv("pantheon_comparison.csv", index=False)

    pd.DataFrame([{
        "phi_0": phi_arr[-1], "w_0": solution["w0"],
        "H_0": solution["H_today"],
        "Omega_m0": Omega_m[-1], "Omega_phi0": Omega_phi[-1],
        "rho_phi0": rho_phi_arr[-1], "V_phi0": potential(phi_arr[-1])
    }]).to_csv("verification_numbers.csv", index=False)

    z_plot = np.linspace(0.0, np.max(z_arr), 200)
    H_interp = interp1d(z_arr, H_arr, kind="linear",
                        bounds_error=False, fill_value="extrapolate")
    H_CFT_plot = H_interp(z_plot)
    H_LCDM_plot = np.sqrt(0.3 * (1.0 + z_plot)**3 + 0.7)
    pd.DataFrame({
        "z": z_plot, "H_CFT": H_CFT_plot, "H_LCDM": H_LCDM_plot,
        "Delta_H_percent": (H_CFT_plot - H_LCDM_plot)/H_LCDM_plot * 100
    }).to_csv("H_comparison.csv", index=False)

    # ------------------------------------------------------------------
    # PUBLICATION‑READY SIX‑PANEL FIGURE
    # ------------------------------------------------------------------
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    axs[0,0].plot(solution["t"], a_arr, "b-")
    axs[0,0].set_xlabel("t")
    axs[0,0].set_ylabel("a")
    axs[0,0].set_title("Scale factor")
    axs[0,0].grid(True)

    axs[0,1].plot(solution["t"], phi_arr, "darkorange")
    axs[0,1].set_xlabel("t")
    axs[0,1].set_ylabel("φ")
    axs[0,1].set_title("Causal field")
    axs[0,1].grid(True)

    # (0,2) EoS – logarithmic redshift axis
    axs[0,2].plot(z_arr, w_phi, "purple")
    axs[0,2].set_xlabel("z")
    axs[0,2].set_ylabel("w_φ")
    axs[0,2].set_title("EoS parameter")
    axs[0,2].grid(True)
    axs[0,2].set_xscale("log")
    axs[0,2].invert_xaxis()
    axs[0,2].set_ylim(-1.5, 0.5)

    if pantheon_available:
        axs[1,0].errorbar(z_sne, mu_obs, yerr=sigma, fmt=".", alpha=0.5,
                          label="Pantheon+")
        axs[1,0].plot(z_sne, mu_th, "r-",
                      label=f"CFT (χ²/dof={chi2_sn/dof:.2f})")
        axs[1,0].legend()
    else:
        axs[1,0].text(0.5, 0.5,
                      "Pantheon+ comparison\n(data unavailable)",
                      transform=axs[1,0].transAxes, ha="center", va="center",
                      fontsize=12)
    axs[1,0].set_xlabel("z")
    axs[1,0].set_ylabel("μ (mag)")
    axs[1,0].set_title("Hubble diagram")
    axs[1,0].grid(True)

    if pantheon_available:
        axs[1,1].axhline(0, color="gray", ls="--")
        axs[1,1].scatter(z_sne, residuals_sn, c="blue", s=20, alpha=0.6)
    else:
        axs[1,1].text(0.5, 0.5,
                      "Residuals\n(data unavailable)",
                      transform=axs[1,1].transAxes, ha="center", va="center",
                      fontsize=12)
    axs[1,1].set_xlabel("z")
    axs[1,1].set_ylabel("Δμ (mag)")
    axs[1,1].set_title("CFT Residuals")
    axs[1,1].grid(True)

    # (1,2) Density parameters – logarithmic redshift axis
    axs[1,2].plot(z_arr, Omega_m, "b-", label="Ω_m")
    axs[1,2].plot(z_arr, Omega_phi, "r-", label="Ω_φ")
    axs[1,2].set_xlabel("z")
    axs[1,2].set_ylabel("Ω")
    axs[1,2].set_title("Density parameters")
    axs[1,2].legend()
    axs[1,2].grid(True)
    axs[1,2].set_xscale("log")
    axs[1,2].invert_xaxis()

    plt.tight_layout()
    plt.savefig("fig_background_evolution.pdf", bbox_inches="tight")
    plt.close()

    print("\n✓ All results generated and saved.")
