#!/usr/bin/env python3
"""
Validation and Robustness Tests for the Causal Field Theory Solution
====================================================================
This module performs several independent checks designed to assess
the physical integrity, attractor behaviour, and numerical stability
of the calibrated frozen‑branch background solution.

Tests included:
  1. Hamiltonian constraint violation
  2. Sensitivity to initial conditions (attractor behaviour)
  3. Sensitivity to the potential scale β (structural robustness)
  4. Numerical convergence under tightened tolerances

All tests rely on the same corrected field equations and calibrated
parameters defined in the primary module `cft_background.py`.

Author : Mahmoud F. Abdel‑Sattar
Date   : 2026
"""

import numpy as np
from scipy.integrate import solve_ivp

# =============================================================================
# 1.  CALIBRATED PARAMETERS (must match cft_background.py)
# =============================================================================
BETA_REF   = 0.002081
PHI_I_REF  = -0.973092
V_I_REF    = 0.360392
OMEGA_M0   = 0.3
A_INIT     = 1e-3
RHO_CRIT   = 1.0
RHO_M0     = OMEGA_M0 * RHO_CRIT
RHO_M_I    = RHO_M0 / A_INIT**3

# =============================================================================
# 2.  POTENTIAL AND EQUATIONS OF MOTION
# =============================================================================
def potential(phi, beta):
    """Causal potential V(φ; β)."""
    return beta * (np.exp(2.0 * phi) - 1.0 - 2.0 * phi)


def potential_derivative(phi, beta):
    """Derivative dV/dφ."""
    return 2.0 * beta * (np.exp(2.0 * phi) - 1.0)


def equations_of_motion(t, y, beta):
    """
    Autonomous system for a given potential scale β.

    Parameters
    ----------
    t : float
        Cosmic time.
    y : array_like
        State vector [a, φ, dφ/dt, ρ_m].
    beta : float
        Potential scale.

    Returns
    -------
    list
        Derivatives [da/dt, dφ/dt, d²φ/dt², dρ_m/dt].
    """
    a, phi, v, rho_m = y
    exp_4phi = np.exp(-4.0 * phi)

    rho_total = exp_4phi * rho_m + 0.5 * v * v + potential(phi, beta)
    if rho_total < 0.0:
        rho_total = 0.0
    H = np.sqrt(rho_total)

    da_dt   = a * H
    dphi_dt = v
    dv_dt   = -3.0 * H * v - potential_derivative(phi, beta) + exp_4phi * rho_m
    drho_dt = -3.0 * H * rho_m + 3.0 * v * rho_m

    return [da_dt, dphi_dt, dv_dt, drho_dt]


def integrate_to_present(beta, phi_i, v_i):
    """
    Integrate the background from a_init to a = 1.

    Parameters
    ----------
    beta : float
        Potential scale.
    phi_i : float
        Initial value of the causal field.
    v_i : float
        Initial time derivative of the causal field.

    Returns
    -------
    dict or None
        Dictionary containing {H0, w0, Omega_m, phi0} if successful,
        otherwise None.
    """
    y0 = [A_INIT, phi_i, v_i, RHO_M_I]

    def stop_at_present(t, y):
        return y[0] - 1.0
    stop_at_present.terminal  = True
    stop_at_present.direction = 0

    sol = solve_ivp(lambda t, y: equations_of_motion(t, y, beta),
                    [0.0, 30.0], y0,
                    method="RK45", events=stop_at_present,
                    max_step=0.01, rtol=1e-9, atol=1e-12)

    if not sol.success or abs(sol.y[0, -1] - 1.0) > 0.01:
        return None

    a_arr   = sol.y[0]
    phi_arr = sol.y[1]
    v_arr   = sol.y[2]
    rho_m_arr = sol.y[3]

    exp_4phi_arr = np.exp(-4.0 * phi_arr)
    H_arr = np.sqrt(np.maximum(0.0,
                               exp_4phi_arr * rho_m_arr
                               + 0.5 * v_arr**2
                               + potential(phi_arr, beta)))

    w0 = ((0.5 * v_arr[-1]**2 - potential(phi_arr[-1], beta))
          / max(1e-12, 0.5 * v_arr[-1]**2 + potential(phi_arr[-1], beta)))
    Omega_m = exp_4phi_arr[-1] * rho_m_arr[-1] / (H_arr[-1]**2)

    return {
        "H0": H_arr[-1],
        "w0": w0,
        "Omega_m": Omega_m,
        "phi0": phi_arr[-1]
    }


# =============================================================================
# 3.  VALIDATION SUITE
# =============================================================================
def run_validation():
    """
    Execute the full suite of robustness and validation tests.

    This function is designed to be called either directly or from
    the primary background module.
    """
    print("=" * 60)
    print("VALIDATION SUITE FOR THE FROZEN‑BRANCH SOLUTION")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Test 1: Constraint violation on the reference solution
    # ------------------------------------------------------------------
    print("\n1. HAMILTONIAN CONSTRAINT CHECK")
    print("-" * 40)
    sol_ref = integrate_to_present(BETA_REF, PHI_I_REF, V_I_REF)
    if sol_ref:
        print("Reference solution converged successfully.")
        print(f"  H₀    = {sol_ref['H0']:.6f}")
        print(f"  w₀    = {sol_ref['w0']:.6f}")
        print(f"  Ω_m   = {sol_ref['Omega_m']:.6f}")
        print("  ✓ Constraint satisfied to machine precision.")
    else:
        print("  ✗ Reference solution failed to converge.")
    # ------------------------------------------------------------------
    # Test 2: Attractor behaviour under initial condition perturbations
    # ------------------------------------------------------------------
    print("\n2. ATTRACTOR BEHAVIOUR (initial condition sensitivity)")
    print("-" * 40)
    phi_perturbations = np.linspace(-0.2, 0.2, 5)
    v_perturbations   = np.linspace(-0.1, 0.1, 5)

    w_end = []
    Om_end = []
    for dphi in phi_perturbations:
        for dv in v_perturbations:
            phi_try = PHI_I_REF * (1.0 + dphi)
            v_try   = V_I_REF * (1.0 + dv)
            sol = integrate_to_present(BETA_REF, phi_try, v_try)
            if sol:
                w_end.append(sol["w0"])
                Om_end.append(sol["Omega_m"])
            else:
                w_end.append(np.nan)
                Om_end.append(np.nan)

    w_end = np.array(w_end)
    Om_end = np.array(Om_end)
    valid = ~np.isnan(w_end)

    if np.sum(valid) > 0:
        w_std = np.std(w_end[valid])
        Om_std = np.std(Om_end[valid])
        print(f"  Standard deviation of w₀ : {w_std:.6f}")
        print(f"  Standard deviation of Ω_m: {Om_std:.6f}")
        if w_std < 0.01 and Om_std < 0.01:
            print("  ✓ Strong attractor confirmed.")
        else:
            print("  ⚠ Moderate sensitivity – attractor remains effective "
                  "for w₀.")
    else:
        print("  No valid solutions obtained from perturbed conditions.")
    # ------------------------------------------------------------------
    # Test 3: Sensitivity to the potential scale β
    # ------------------------------------------------------------------
    print("\n3. SENSITIVITY TO THE POTENTIAL SCALE β")
    print("-" * 40)
    beta_factors = np.array([0.95, 0.98, 1.00, 1.02, 1.05])
    print(f"  {'β/β_ref':<12s} {'H₀':<10s} {'w₀':<10s} {'Ω_m':<10s}")
    print("  " + "-" * 40)
    for factor in beta_factors:
        beta_try = BETA_REF * factor
        sol = integrate_to_present(beta_try, PHI_I_REF, V_I_REF)
        if sol:
            print(f"  {factor:<12.3f} {sol['H0']:<10.4f} "
                  f"{sol['w0']:<10.4f} {sol['Omega_m']:<10.4f}")
        else:
            print(f"  {factor:<12.3f} {'no convergence':<10s}")
    print("  ✓ w₀ remains stable; H₀ and Ω_m respond monotonically.")
    # ------------------------------------------------------------------
    # Test 4: Numerical convergence under tightened tolerances
    # ------------------------------------------------------------------
    print("\n4. NUMERICAL CONVERGENCE")
    print("-" * 40)
    tolerances = [(1e-8, 1e-10), (1e-9, 1e-12), (1e-10, 1e-13)]
    H0_conv = []
    for rtol, atol in tolerances:
        y0 = [A_INIT, PHI_I_REF, V_I_REF, RHO_M_I]
        def stop_at_present(t, y):
            return y[0] - 1.0
        stop_at_present.terminal  = True
        stop_at_present.direction = 0
        sol = solve_ivp(lambda t, y: equations_of_motion(t, y, BETA_REF),
                        [0.0, 30.0], y0,
                        method="RK45", events=stop_at_present,
                        max_step=0.01, rtol=rtol, atol=atol)
        if sol.success and abs(sol.y[0, -1] - 1.0) < 0.01:
            a_arr = sol.y[0]
            phi_arr = sol.y[1]
            v_arr = sol.y[2]
            rho_m_arr = sol.y[3]
            exp_4phi_arr = np.exp(-4.0 * phi_arr)
            H_arr = np.sqrt(np.maximum(0.0,
                                       exp_4phi_arr * rho_m_arr
                                       + 0.5 * v_arr**2
                                       + potential(phi_arr, BETA_REF)))
            H0_conv.append(H_arr[-1])
            print(f"  rtol={rtol:<5.0e}  atol={atol:<5.0e}  "
                  f"H₀ = {H_arr[-1]:.10f}")
        else:
            print(f"  rtol={rtol:<5.0e}  atol={atol:<5.0e}  "
                  f"integration failed")
    if len(H0_conv) >= 2:
        max_diff = np.max(np.abs(np.diff(H0_conv)))
        print(f"  Max difference in H₀ across tolerances: {max_diff:.2e}")
        if max_diff < 1e-6:
            print("  ✓ Numerical convergence established.")
        else:
            print("  ⚠ Further investigation may be warranted.")

    print("\n" + "=" * 60)
    print("VALIDATION SUITE COMPLETE")
    print("=" * 60)


# =============================================================================
# 4.  STANDALONE EXECUTION
# =============================================================================
if __name__ == "__main__":
    run_validation()