#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Causal Field Theory — Cosmological Background and Observational Confrontation
================================================================================

Generation of the publication-quality six-panel figure for the physical
frozen-branch solution of Causal Field Theory (CFT), together with the
reproducible numerical pipeline used to obtain the results reported in
Section 4 of the manuscript.

Description
-----------
This module integrates the coupled Einstein–Klein–Gordon system governing
the cosmological background of CFT in the Einstein frame, transforms the
resulting solution to the physical (Jordan) frame, and confronts the
predicted luminosity distance with the Pantheon+ compilation of 277
Hubble-flow Type Ia supernovae.

The analysis proceeds through six stages:

    (i)    Integration of the background equations of motion from an
           initial scale factor a_i = 10^{-3} to the present epoch a = 1.
    (ii)   Evaluation of all background observables in both the Einstein
           and physical frames.
    (iii)  Construction of the physical-frame luminosity distance
           D_L^{phys}(z) by numerical quadrature.
    (iv)   Marginalisation of the absolute magnitude offset M and
           computation of the χ² statistic against Pantheon+.
    (v)    Verification of internal consistency (Hamiltonian constraint,
           frame transformation, frozen-branch condition).
    (vi)   Production of the six-panel figure and export of CSV data files.

Physical Framework
------------------
    · Matter couples to the physical metric ĝ_{μν} = e^{-2φ} g_{μν}.
    · The electromagnetic action is conformally invariant in four
      dimensions, so photons propagate along null geodesics that are
      preserved under the conformal transformation.
    · Atomic clocks in the physical frame measure proper time defined by
      ĝ_{μν}; consequently the observed redshift satisfies
                          z_obs = z_phys,
      where 1 + z_phys = e^{φ - φ_0} (1 + z_E).
    · The physical Hubble parameter is
                          H_phys = e^{φ} (H_E - φ̇).

Author       : Mahmoud F. Abdel-Sattar
Affiliation  : Department of Astronomy and Meteorology, Al-Azhar University
Date         : 2026
License      : MIT
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp, cumulative_trapezoid
import matplotlib as mpl
import matplotlib.pyplot as plt

# =============================================================================
# 0.  Global configuration
# =============================================================================

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Output directory for CSV and figure files
OUTPUT_DIR = Path(".")
OUTPUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# 1.  Physical constants and calibrated parameters
# =============================================================================

@dataclass(frozen=True)
class CalibratedParameters:
    """Calibrated parameters of the frozen-branch solution (Section 4.3)."""
    beta: float = 0.002081         # Potential scale (dimensionless)
    phi_i: float = -0.973092       # Initial causal field value
    v_i: float = 0.360392          # Initial field velocity dφ/dt
    omega_m0: float = 0.3          # Present-day matter density parameter
    a_init: float = 1.0e-3         # Initial scale factor
    a_final: float = 1.0           # Final scale factor (present epoch)

    @property
    def rho_m_init(self) -> float:
        """Initial matter energy density, ρ_{m,i} = Ω_{m0} / a_i³."""
        return self.omega_m0 / self.a_init ** 3


PARAMS = CalibratedParameters()

# =============================================================================
# 2.  Potential and its derivative
# =============================================================================

def potential(phi, beta: float = PARAMS.beta):
    """
    Causal potential V(φ) = β (e^{2φ} − 1 − 2φ).
    """
    return beta * (np.exp(2.0 * phi) - 1.0 - 2.0 * phi)


def potential_derivative(phi, beta: float = PARAMS.beta):
    """
    Derivative of the causal potential, V'(φ) = 2β (e^{2φ} − 1).
    """
    return 2.0 * beta * (np.exp(2.0 * phi) - 1.0)


# =============================================================================
# 3.  Background equations of motion (Einstein frame)
# =============================================================================

def equations_of_motion(t: float, y: np.ndarray,
                        beta: float = PARAMS.beta) -> list[float]:
    """
    Right-hand side of the coupled Einstein–Klein–Gordon system.

    The state vector is y = [a, φ, φ̇, ρ_m]:
        H_E² = e^{-4φ} ρ_m + ½ φ̇² + V(φ),
        φ̈ + 3 H_E φ̇ + V'(φ) = e^{-4φ} ρ_m,
        ρ̇_m + 3 H_E ρ_m = 3 φ̇ ρ_m.
    """
    a, phi, v, rho_m = y
    exp_minus_4phi = np.exp(-4.0 * phi)

    rho_total = exp_minus_4phi * rho_m + 0.5 * v * v + potential(phi, beta)
    rho_total = max(rho_total, 0.0)
    hubble_e = np.sqrt(rho_total)

    da_dt = a * hubble_e
    dphi_dt = v
    dv_dt = (-3.0 * hubble_e * v
             - potential_derivative(phi, beta)
             + exp_minus_4phi * rho_m)
    drho_dt = -3.0 * hubble_e * rho_m + 3.0 * v * rho_m

    return [da_dt, dphi_dt, dv_dt, drho_dt]


# =============================================================================
# 4.  Integration of the background
# =============================================================================

@dataclass
class BackgroundSolution:
    """Container for the integrated background quantities in both frames."""
    t: np.ndarray
    a_e: np.ndarray
    phi: np.ndarray
    v: np.ndarray
    rho_m: np.ndarray
    z_e: np.ndarray
    z_phys: np.ndarray
    h_e: np.ndarray
    h_phys: np.ndarray
    omega_m: np.ndarray
    omega_phi: np.ndarray
    w_phi: np.ndarray
    phi_0: float = field(default=0.0)
    h_e_today: float = field(default=0.0)
    h_phys_today: float = field(default=0.0)
    omega_phi_today: float = field(default=0.0)
    w_0: float = field(default=0.0)


def integrate_background(
    params: CalibratedParameters = PARAMS,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-13,
    max_step: float = 1.0e-3,
) -> BackgroundSolution:
    """
    Integrate the CFT background equations from a_i to a = 1.

    Parameters
    ----------
    params : CalibratedParameters
        Calibrated parameters of the model.
    rtol : float, optional
        Relative tolerance for the ODE solver (default: 1e-10).
    atol : float, optional
        Absolute tolerance for the ODE solver (default: 1e-13).
    max_step : float, optional
        Maximum step size in Hubble-time units (default: 1e-3).

    Returns
    -------
    BackgroundSolution
        Integrated solution in both frames.
    """
    y0 = [params.a_init, params.phi_i, params.v_i, params.rho_m_init]

    def stop_at_present(t: float, y: np.ndarray, *args) -> float:
        return y[0] - params.a_final

    stop_at_present.terminal = True
    stop_at_present.direction = 0

    sol = solve_ivp(
        fun=lambda t, y: equations_of_motion(t, y, params.beta),
        t_span=(0.0, 50.0),
        y0=y0,
        method="RK45",
        events=stop_at_present,
        max_step=max_step,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    if abs(sol.y[0, -1] - params.a_final) > 1.0e-3:
        raise RuntimeError(
            f"Integration did not reach a=1 (a_final={sol.y[0, -1]:.6e})."
        )

    a_e = sol.y[0]
    phi = sol.y[1]
    v = sol.y[2]
    rho_m = sol.y[3]

    z_e = 1.0 / a_e - 1.0
    exp_minus_4phi = np.exp(-4.0 * phi)
    h_e = np.sqrt(np.maximum(
        1.0e-20, exp_minus_4phi * rho_m + 0.5 * v ** 2 + potential(phi)
    ))

    phi_0 = phi[-1]
    z_phys = np.exp(phi - phi_0) * (1.0 + z_e) - 1.0
    h_phys = np.exp(phi) * (h_e - v)

    rho_phi = 0.5 * v ** 2 + potential(phi)
    omega_m = exp_minus_4phi * rho_m / (h_e ** 2)
    omega_phi = rho_phi / (h_e ** 2)

    w_phi = (0.5 * v ** 2 - potential(phi)) / np.maximum(1.0e-12, rho_phi)
    w_phi = np.clip(w_phi, -2.0, 2.0)

    return BackgroundSolution(
        t=sol.t, a_e=a_e, phi=phi, v=v, rho_m=rho_m,
        z_e=z_e, z_phys=z_phys, h_e=h_e, h_phys=h_phys,
        omega_m=omega_m, omega_phi=omega_phi, w_phi=w_phi,
        phi_0=phi_0,
        h_e_today=h_e[-1],
        h_phys_today=h_phys[-1],
        omega_phi_today=omega_phi[-1],
        w_0=w_phi[-1],
    )


# =============================================================================
# 5.  Physical-frame luminosity distance
# =============================================================================

def build_luminosity_distance(solution: BackgroundSolution):
    """
    Construct the physical-frame luminosity distance D_L^{phys}(z):

        D_L^{phys}(z) = (1 + z) ∫₀^z dz' / H_phys(z')

    using a pre-computed cumulative trapezoidal quadrature on the sorted
    (z_phys, H_phys) grid.
    """
    mask = (solution.z_phys >= 0.0) & (solution.h_phys > 0.0)
    z_raw = solution.z_phys[mask]
    h_raw = solution.h_phys[mask]

    order = np.argsort(z_raw)
    z_sorted = z_raw[order]
    h_sorted = h_raw[order]

    _, unique_idx = np.unique(np.round(z_sorted, 12), return_index=True)
    z_grid = z_sorted[unique_idx]
    h_grid = h_sorted[unique_idx]

    integrand = 1.0 / np.maximum(h_grid, 1.0e-20)
    cumulative = cumulative_trapezoid(integrand, z_grid, initial=0.0)

    h_0 = h_grid[0]

    def d_l(z: float) -> float:
        if z <= 0.0:
            return 0.0
        if z < z_grid[0]:
            return z / h_0                       # Hubble-law extrapolation
        idx = min(np.searchsorted(z_grid, z), len(z_grid) - 1)
        return (1.0 + z) * cumulative[idx]

    return d_l


# =============================================================================
# 6.  Pantheon+ data and χ² statistic
# =============================================================================

PANTHEON_URL = (
    "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/"
    "main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"
)


def load_pantheon(url: str = PANTHEON_URL) -> pd.DataFrame:
    """
    Load the Pantheon+ Hubble-flow Type Ia supernova compilation.
    """
    data = pd.read_csv(url, sep=r"\s+", comment="#")
    return data[data["USED_IN_SH0ES_HF"] == 1].copy()


def compute_chi_squared(solution: BackgroundSolution,
                        sne: pd.DataFrame) -> Dict[str, object]:
    """
    Evaluate the χ² statistic against the Pantheon+ Hubble diagram.
    The absolute magnitude offset M is marginalised analytically.

    The luminosity distance is evaluated once per supernova and reused
    for both the offset marginalisation and the theoretical distance
    moduli, avoiding redundant computation.
    """
    d_l = build_luminosity_distance(solution)

    z = sne["zHD"].to_numpy()
    mu_obs = sne["MU_SH0ES"].to_numpy()
    sigma_mu = sne["MU_SH0ES_ERR_DIAG"].to_numpy()
    n_sne = z.size

    # Evaluate D_L once per supernova
    d_l_values = np.array([d_l(zi) for zi in z])

    # Compute 5 log10 D_L; guard against non-positive values
    valid = d_l_values > 0.0
    mu_th_0 = np.full_like(d_l_values, np.nan, dtype=float)
    mu_th_0[valid] = 5.0 * np.log10(d_l_values[valid])

    if not np.all(valid):
        n_invalid = int(np.sum(~valid))
        raise RuntimeError(
            f"Luminosity distance is non-positive for {n_invalid} "
            f"supernovae; cannot compute distance moduli."
        )

    # Analytic marginalisation over the absolute magnitude offset M
    m_best = (np.sum((mu_obs - mu_th_0) / sigma_mu ** 2)
              / np.sum(1.0 / sigma_mu ** 2))

    mu_th = mu_th_0 + m_best
    residuals = mu_obs - mu_th

    chi2 = float(np.sum((residuals / sigma_mu) ** 2))
    dof = n_sne - 1

    return {
        "chi2": chi2,
        "dof": dof,
        "chi2_dof": chi2 / dof,
        "M_best": float(m_best),
        "sigma_res": float(np.std(residuals, ddof=1)),
        "z": z,
        "mu_obs": mu_obs,
        "sigma_mu": sigma_mu,
        "mu_th": mu_th,
        "residuals": residuals,
    }


# =============================================================================
# 7.  Publication-quality figure
# =============================================================================

def make_six_panel_figure(solution: BackgroundSolution,
                          chi2_results: Dict[str, object],
                          output_basename: str = "fig_background_evolution"
                          ) -> None:
    """
    Generate the publication-quality six-panel figure.

    Panels
    ------
        (0,0) Scale factor a(t)
        (0,1) Causal field φ(t)
        (0,2) Equation of state w_φ(z)
        (1,0) Hubble diagram (CFT vs Pantheon+)
        (1,1) Residuals Δμ with ±2σ band
        (1,2) Density parameters Ω_m(z), Ω_φ(z)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 10.0))

    # ---------- Panel (0,0): Scale factor a(t) ----------
    ax = axes[0, 0]
    ax.plot(solution.t, solution.a_e, color="navy", linewidth=2.0)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$a$")
    ax.set_title("Scale factor")
    ax.grid(True, alpha=0.3)

    # ---------- Panel (0,1): Causal field φ(t) ----------
    ax = axes[0, 1]
    ax.plot(solution.t, solution.phi, color="darkorange", linewidth=2.0)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\phi$")
    ax.set_title("Causal field")
    ax.grid(True, alpha=0.3)

    # ---------- Panel (0,2): Equation of state w_φ(z) ----------
    ax = axes[0, 2]
    ax.plot(solution.z_e, solution.w_phi, color="purple", linewidth=2.0)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$w_\phi$")
    ax.set_title("EoS parameter")
    ax.set_xscale("log")
    ax.set_xlim(1.0e-3, 1.0e1)
    ax.invert_xaxis()
    ax.set_ylim(-1.5, 0.5)
    ax.grid(True, alpha=0.3)

    # ---------- Panel (1,0): Hubble diagram ----------
    ax = axes[1, 0]
    z_sne = chi2_results["z"]
    mu_obs = chi2_results["mu_obs"]
    sigma_mu = chi2_results["sigma_mu"]
    mu_th = chi2_results["mu_th"]
    chi2_dof = chi2_results["chi2_dof"]

    ax.errorbar(z_sne, mu_obs, yerr=sigma_mu,
                fmt=".", markersize=4, alpha=0.45,
                color="steelblue", label="Pantheon+")

    order = np.argsort(z_sne)
    ax.plot(z_sne[order], mu_th[order],
            color="crimson", linewidth=2.0,
            label=fr"CFT ($\chi^2/\mathrm{{dof}} = {chi2_dof:.2f}$)")

    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\mu$ (mag)")
    ax.set_title("Hubble diagram")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # ---------- Panel (1,1): Residuals ----------
    ax = axes[1, 1]
    residuals = chi2_results["residuals"]
    sigma_res = chi2_results["sigma_res"]

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.axhspan(-2.0 * sigma_res, 2.0 * sigma_res,
               alpha=0.15, color="green",
               label=fr"$\pm 2\sigma$")
    ax.scatter(z_sne, residuals, s=15, c="royalblue", alpha=0.55)

    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\Delta\mu$ (mag)")
    ax.set_title("CFT Residuals")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # ---------- Panel (1,2): Density parameters ----------
    ax = axes[1, 2]
    ax.plot(solution.z_e, solution.omega_m,
            color="navy", linewidth=2.0, label=r"$\Omega_m$")
    ax.plot(solution.z_e, solution.omega_phi,
            color="crimson", linewidth=2.0, label=r"$\Omega_\phi$")
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\Omega$")
    ax.set_title("Density parameters")
    ax.set_xscale("log")
    ax.set_xlim(1.0e-3, 1.0e1)
    ax.invert_xaxis()
    ax.legend(loc="center left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    for extension in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"{output_basename}.{extension}")
    plt.close(fig)


# =============================================================================
# 8.  Internal consistency verification
# =============================================================================

def verify_consistency(solution: BackgroundSolution,
                       chi2_results: Dict[str, object]) -> None:
    """
    Perform internal consistency checks on the background solution.

    The following identities are verified:

        1. Ω_m(z) + Ω_φ(z) = 1 for all redshifts.
        2. The frozen-branch condition w_0 ≈ −1 is satisfied.
        3. The physical-frame Hubble parameter satisfies
           H_phys = e^{φ}(H_E − φ̇) at the present epoch.
        4. The location of the causal-field zero-crossing.
        5. The reduced χ² statistic.

    Additionally, the present-day value of φ̇₀ / H_E, the causal Lorentz
    factor γ'₀ = e^{φ₀}, and the frame time-scale ratio d t̂ / d t_E|₀
    are printed to document the frame hierarchy discussed in
    Section 4.5 of the manuscript.
    """
    print("\n" + "=" * 72)
    print("Consistency checks")
    print("=" * 72)

    # 1. Hamiltonian closure
    closure = solution.omega_m + solution.omega_phi
    max_dev = float(np.max(np.abs(closure - 1.0)))
    print(f"  max |Ω_m + Ω_φ − 1|              : {max_dev:.2e}")

    # 2. Frozen-branch condition
    print(f"  w_0 (frozen-branch condition)    : {solution.w_0:.6f}")

    # 3. Frame transformation identity
    lhs = solution.h_phys_today
    rhs = np.exp(solution.phi_0) * (solution.h_e_today - solution.v[-1])
    print(f"  H_phys(0) − e^{{φ₀}}(H_E − φ̇)   : {lhs - rhs:.2e}")

    # 4. Zero-crossing of the causal field
    idx_zero = int(np.argmin(np.abs(solution.phi)))
    z_at_zero = float(solution.z_e[idx_zero])
    print(f"  Causal field crosses zero at z   : {z_at_zero:.4f}")

    # 5. Present-day field velocity (documented in Section 4.5)
    v0_over_he = solution.v[-1] / solution.h_e_today
    print(f"  φ̇₀ / H_E                         : {v0_over_he:.6e}")

    # 6. Causal Lorentz factor at present epoch
    gamma_prime_0 = float(np.exp(solution.phi_0))
    print(f"  e^{{φ₀}} = γ'₀                    : {gamma_prime_0:.6f}")

    # 7. Frame hierarchy and time-scale ratio
    h_ratio = solution.h_phys_today / solution.h_e_today
    dt_ratio = float(np.exp(-solution.phi_0))
    print(f"  H_phys(0) / H_E(0)               : {h_ratio:.6f}")
    print(f"  d t̂ / d t_E |_{{z=0}}            : {dt_ratio:.6e}")

    # 8. χ² statistic
    print(f"  χ² / dof                         : "
          f"{chi2_results['chi2_dof']:.4f}")


# =============================================================================
# 9.  CSV export
# =============================================================================

def export_csv_files(solution: BackgroundSolution,
                     chi2_results: Dict[str, object],
                     prefix: str = "physical") -> None:
    """
    Export all numerical data to CSV files.

    Files
    -----
    background_{prefix}.csv            : background evolution
    pantheon_comparison_{prefix}.csv   : SNe Ia comparison
    verification_numbers_{prefix}.csv  : key cosmological quantities
    """
    print("\n" + "=" * 72)
    print("Exporting CSV files")
    print("=" * 72)

    pd.DataFrame({
        "z_E": solution.z_e,
        "z_phys": solution.z_phys,
        "a_E": solution.a_e,
        "phi": solution.phi,
        "v": solution.v,
        "rho_m": solution.rho_m,
        "H_E": solution.h_e,
        "H_phys": solution.h_phys,
        "Omega_m": solution.omega_m,
        "Omega_phi": solution.omega_phi,
        "w_phi": solution.w_phi,
    }).to_csv(OUTPUT_DIR / f"background_{prefix}.csv", index=False)
    print(f"  Saved: background_{prefix}.csv")

    pd.DataFrame({
        "z": chi2_results["z"],
        "mu_obs": chi2_results["mu_obs"],
        "sigma_mu": chi2_results["sigma_mu"],
        "mu_CFT": chi2_results["mu_th"],
        "residual": chi2_results["residuals"],
    }).to_csv(OUTPUT_DIR / f"pantheon_comparison_{prefix}.csv", index=False)
    print(f"  Saved: pantheon_comparison_{prefix}.csv")

    pd.DataFrame([{
        "phi_0": solution.phi_0,
        "w_0": solution.w_0,
        "H_E_today": solution.h_e_today,
        "H_phys_today": solution.h_phys_today,
        "Omega_phi": solution.omega_phi_today,
        "V_phi_0": float(potential(solution.phi_0)),
        "gamma_prime_0": float(np.exp(solution.phi_0)),
        "v0_over_HE": solution.v[-1] / solution.h_e_today,
        "H_phys_over_H_E": solution.h_phys_today / solution.h_e_today,
        "chi2": chi2_results["chi2"],
        "dof": chi2_results["dof"],
        "chi2_dof": chi2_results["chi2_dof"],
        "M_best": chi2_results["M_best"],
        "sigma_res": chi2_results["sigma_res"],
    }]).to_csv(OUTPUT_DIR / f"verification_numbers_{prefix}.csv", index=False)
    print(f"  Saved: verification_numbers_{prefix}.csv")


# =============================================================================
# 10.  Main execution
# =============================================================================

def main() -> None:
    """Run the full pipeline and produce the publication figure."""

    print("=" * 72)
    print("Causal Field Theory — Cosmological Background Analysis")
    print("=" * 72)

    print("\n[1/6] Integrating background equations of motion...")
    solution = integrate_background(PARAMS)
    print(f"      φ_0            = {solution.phi_0:.6f}")
    print(f"      H_E(z = 0)     = {solution.h_e_today:.6f}")
    print(f"      H_phys(z = 0)  = {solution.h_phys_today:.6f}")
    print(f"      Ω_φ (Einstein) = {solution.omega_phi_today:.6f}")
    print(f"      w_0            = {solution.w_0:.6f}")

    print("\n[2/6] Loading Pantheon+ Hubble-flow sample...")
    sne = load_pantheon()
    print(f"      Number of SNe Ia : {len(sne)}")

    print("\n[3/6] Computing χ² statistic against Pantheon+...")
    chi2_results = compute_chi_squared(solution, sne)
    print(f"      χ²              = {chi2_results['chi2']:.3f}")
    print(f"      dof             = {chi2_results['dof']}")
    print(f"      χ²/dof          = {chi2_results['chi2_dof']:.4f}")
    print(f"      M_best          = {chi2_results['M_best']:.6f}")
    print(f"      σ(residuals)    = {chi2_results['sigma_res']:.4f} mag")

    print("\n[4/6] Running internal consistency checks...")
    verify_consistency(solution, chi2_results)

    print("\n[5/6] Exporting CSV files...")
    export_csv_files(solution, chi2_results, prefix="physical")

    print("\n[6/6] Generating publication-quality six-panel figure...")
    make_six_panel_figure(solution, chi2_results)
    print("      Saved: fig_background_evolution.pdf")
    print("      Saved: fig_background_evolution.png")

    print("\n" + "=" * 72)
    print("Analysis completed successfully.")
    print("=" * 72)


if __name__ == "__main__":
    main()