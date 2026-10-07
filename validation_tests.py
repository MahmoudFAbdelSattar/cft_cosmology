#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Causal Field Theory — Validation Test Suite
================================================================================

Independent robustness checks of the frozen-branch solution of Causal
Field Theory, as described in Appendix A.4 of the manuscript.  The tests
are executed without re-calibrating the model and include:

    1. Hamiltonian constraint      (to machine precision)
    2. Sensitivity to β            (±5% variation of the potential scale)
    3. Numerical convergence       (tightened ODE tolerances)

The script is designed to be self-contained and can be executed either as
a standalone module that imports from `cft_background.py`, or directly
inside a Google Colab cell (in which case the required definitions are
provided locally if the import fails).

Usage
-----
    $ python validation_tests.py

Author       : Mahmoud F. Abdel-Sattar
Affiliation  : Department of Astronomy and Meteorology, Al-Azhar University
Date         : 2026
License      : MIT
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Tuple

import numpy as np

# =============================================================================
# Try to import from cft_background.py.  If unavailable (e.g., when this
# script is executed inside a Colab cell without the companion module on
# disk), fall back to local self-contained definitions.
# =============================================================================

try:
    from cft_background import (
        CalibratedParameters,
        PARAMS,
        equations_of_motion,
        integrate_background,
        potential,
        potential_derivative,
    )
    _STANDALONE = False
except ImportError:
    _STANDALONE = True

    from scipy.integrate import solve_ivp

    @dataclass(frozen=True)
    class CalibratedParameters:
        beta: float = 0.002081
        phi_i: float = -0.973092
        v_i: float = 0.360392
        omega_m0: float = 0.3
        a_init: float = 1.0e-3
        a_final: float = 1.0

        @property
        def rho_m_init(self) -> float:
            return self.omega_m0 / self.a_init ** 3

    PARAMS = CalibratedParameters()

    def potential(phi, beta: float = PARAMS.beta):
        return beta * (np.exp(2.0 * phi) - 1.0 - 2.0 * phi)

    def potential_derivative(phi, beta: float = PARAMS.beta):
        return 2.0 * beta * (np.exp(2.0 * phi) - 1.0)

    def equations_of_motion(t, y, beta=PARAMS.beta):
        a, phi, v, rho_m = y
        e4 = np.exp(-4.0 * phi)
        rho_tot = e4 * rho_m + 0.5 * v * v + potential(phi, beta)
        rho_tot = max(rho_tot, 0.0)
        H_E = np.sqrt(rho_tot)
        return [
            a * H_E,
            v,
            -3.0 * H_E * v - potential_derivative(phi, beta) + e4 * rho_m,
            -3.0 * H_E * rho_m + 3.0 * v * rho_m,
        ]

    def integrate_background(params=PARAMS, rtol=1.0e-10, atol=1.0e-13,
                             max_step=1.0e-3):
        y0 = [params.a_init, params.phi_i, params.v_i, params.rho_m_init]

        def stop(t, y, *args):
            return y[0] - params.a_final
        stop.terminal = True
        stop.direction = 0

        sol = solve_ivp(
            fun=lambda t, y: equations_of_motion(t, y, params.beta),
            t_span=(0.0, 50.0),
            y0=y0,
            method="RK45",
            events=stop,
            max_step=max_step,
            rtol=rtol,
            atol=atol,
        )

        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")
        if abs(sol.y[0, -1] - params.a_final) > 1.0e-3:
            raise RuntimeError(
                f"Did not reach a=1 (a_final={sol.y[0,-1]:.6e})."
            )

        a_e = sol.y[0]
        phi = sol.y[1]
        v = sol.y[2]
        rho_m = sol.y[3]

        e4 = np.exp(-4.0 * phi)
        H_E = np.sqrt(np.maximum(
            1e-20, e4 * rho_m + 0.5 * v ** 2 + potential(phi, params.beta)
        ))

        phi_0 = phi[-1]
        z_E = 1.0 / a_e - 1.0
        z_phys = np.exp(phi - phi_0) * (1.0 + z_E) - 1.0
        H_phys = np.exp(phi) * (H_E - v)

        rho_phi = 0.5 * v ** 2 + potential(phi, params.beta)
        Omega_m = e4 * rho_m / (H_E ** 2)
        Omega_phi = rho_phi / (H_E ** 2)
        w_phi = ((0.5 * v ** 2 - potential(phi, params.beta))
                 / np.maximum(1e-12, rho_phi))
        w_phi = np.clip(w_phi, -2.0, 2.0)

        return {
            "t": sol.t, "a_e": a_e, "phi": phi, "v": v, "rho_m": rho_m,
            "z_E": z_E, "z_phys": z_phys, "H_E": H_E, "H_phys": H_phys,
            "Omega_m": Omega_m, "Omega_phi": Omega_phi, "w_phi": w_phi,
            "phi_0": phi_0,
            "H_E_today": H_E[-1],
            "H_phys_today": H_phys[-1],
            "Omega_phi_today": Omega_phi[-1],
            "w_0": w_phi[-1],
        }


# =============================================================================
# Helper: extract a scalar from either a dict or a dataclass
# =============================================================================

def _get(sol, key):
    """Return sol[key] for dict-like solutions or getattr(sol, key)."""
    if isinstance(sol, dict):
        return sol[key]
    return getattr(sol, key)


# =============================================================================
# Test 1: Hamiltonian constraint
# =============================================================================

def test_hamiltonian_constraint(
    params: CalibratedParameters = PARAMS,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-13,
    verbose: bool = True,
) -> float:
    """
    Verify that the Friedmann constraint is satisfied to machine precision.

    The constraint is
        H_E² = e^{-4φ} ρ_m + ½ φ̇² + V(φ),
    which is satisfied identically by construction of the ODE system.
    The test recomputes both sides at every integration step and returns
    the maximum relative deviation.
    """
    sol = integrate_background(params, rtol=rtol, atol=atol)

    phi = _get(sol, "phi")
    v = _get(sol, "v")
    rho_m = _get(sol, "rho_m")
    H_E = _get(sol, "H_E")

    rhs = np.exp(-4.0 * phi) * rho_m + 0.5 * v ** 2 + potential(phi, params.beta)
    lhs = H_E ** 2

    rel_dev = float(np.max(np.abs(lhs - rhs) / np.maximum(np.abs(rhs), 1e-30)))

    if verbose:
        print("=" * 72)
        print("Test 1 — Hamiltonian constraint")
        print("=" * 72)
        print(f"  Maximum relative deviation : {rel_dev:.3e}")
        if rel_dev < 1.0e-12:
            print("  ✓ Constraint satisfied to machine precision")
        else:
            print("  ⚠ Deviation larger than expected")

    return rel_dev


# =============================================================================
# Test 2: Sensitivity to β
# =============================================================================

def test_beta_sensitivity(
    beta_0: float = PARAMS.beta,
    delta: float = 0.05,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Assess the sensitivity of present-day observables to variations of the
    potential scale β.

    Returns
    -------
    factors : ndarray
        Multiplicative factors applied to β (1 - δ, 1, 1 + δ).
    w0 : ndarray
        Present-day equation of state.
    omega_phi : ndarray
        Present-day dark-energy density parameter.
    h_e : ndarray
        Present-day Einstein-frame Hubble parameter.
    """
    factors = np.array([1.0 - delta, 1.0, 1.0 + delta])
    w0 = np.zeros_like(factors)
    omega_phi = np.zeros_like(factors)
    h_e = np.zeros_like(factors)

    for i, f in enumerate(factors):
        params = replace(PARAMS, beta=beta_0 * f)
        sol = integrate_background(params)
        w0[i] = _get(sol, "w_0")
        omega_phi[i] = _get(sol, "Omega_phi_today")
        h_e[i] = _get(sol, "H_E_today")

    if verbose:
        print("\n" + "=" * 72)
        print("Test 2 — Sensitivity to β")
        print("=" * 72)
        for f, w, om, h in zip(factors, w0, omega_phi, h_e):
            print(f"  β = {beta_0 * f:.6f} ({f * 100:5.0f}%): "
                  f"w_0 = {w:.6f},  Ω_φ = {om:.6f},  H_E(0) = {h:.6f}")
        spread = float(np.max(w0) - np.min(w0))
        print(f"\n  Spread in w_0 : {spread:.3e}")
        if spread < 1.0e-3:
            print("  ✓ w_0 stable to ~3 decimal places under ±5% variation of β")
        else:
            print("  ⚠ w_0 varies with β")

    return factors, w0, omega_phi, h_e


# =============================================================================
# Test 3: Numerical convergence
# =============================================================================

def test_numerical_convergence(
    params: CalibratedParameters = PARAMS,
    tolerances: Tuple[Tuple[float, float], ...] = (
        (1.0e-10, 1.0e-13),
        (1.0e-11, 1.0e-14),
        (1.0e-12, 1.0e-15),
    ),
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Verify that the solution converges as the ODE tolerances are tightened.

    Returns
    -------
    h_today : ndarray
        Present-day Hubble parameter at each tolerance level.
    w0 : ndarray
        Present-day equation of state at each tolerance level.
    """
    h_today = np.zeros(len(tolerances))
    w0 = np.zeros(len(tolerances))

    for i, (rtol, atol) in enumerate(tolerances):
        sol = integrate_background(params, rtol=rtol, atol=atol)
        h_today[i] = _get(sol, "H_E_today")
        w0[i] = _get(sol, "w_0")

    h_diff = np.abs(np.diff(h_today))
    max_diff = float(h_diff.max()) if h_diff.size else 0.0

    if verbose:
        print("\n" + "=" * 72)
        print("Test 3 — Numerical convergence")
        print("=" * 72)
        for i, (rtol, atol) in enumerate(tolerances):
            print(f"  rtol = {rtol:.0e}, atol = {atol:.0e}: "
                  f"H_E(0) = {h_today[i]:.10f},  w_0 = {w0[i]:.10f}")
        print(f"\n  Max |ΔH_E| between consecutive tolerances : {max_diff:.3e}")
        if max_diff < 1.0e-7:
            print("  ✓ Numerical convergence established")
        else:
            print("  ⚠ Convergence not yet achieved")

    return h_today, w0


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    """Execute the full validation suite."""

    print("=" * 72)
    print("Causal Field Theory — Validation Suite")
    print("=" * 72)
    if _STANDALONE:
        print("  (running in standalone mode: local definitions in use)")
    else:
        print("  (importing from cft_background.py)")

    # Reference solution
    print("\n[0/3] Integrating the calibrated reference solution...")
    sol_ref = integrate_background(PARAMS)
    print(f"       φ_0            = {_get(sol_ref, 'phi_0'):.6f}")
    print(f"       H_E(z = 0)     = {_get(sol_ref, 'H_E_today'):.6f}")
    print(f"       Ω_φ (Einstein) = {_get(sol_ref, 'Omega_phi_today'):.6f}")
    print(f"       w_0            = {_get(sol_ref, 'w_0'):.6f}")

    # Test 1
    test_hamiltonian_constraint()

    # Test 2
    test_beta_sensitivity()

    # Test 3
    test_numerical_convergence()

    print("\n" + "=" * 72)
    print("All validation tests completed.")
    print("=" * 72)


if __name__ == "__main__":
    main()