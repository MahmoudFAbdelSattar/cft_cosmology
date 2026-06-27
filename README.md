# Causal Field Theory – Frozen‑Branch Cosmological Solution

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

## Overview

This repository contains the numerical code that accompanies the paper

> **Causal Field Theory: Dynamical Dark Energy from the Principle of Causal Optimality**  
> *Mahmoud F. Abdel‑Sattar* (2026)

The code integrates the corrected cosmological background equations of
**Causal Field Theory (CFT)**, calibrates the model parameters to reproduce
the present‑day Hubble constant, matter density, and an equation of state
\(w_0 = -1\), and confronts the resulting expansion history with the
Pantheon+ compilation of type Ia supernovae.

---

## Theoretical Framework (Summary)

The theory is formulated in the Einstein frame, where matter couples to
a conformally related physical metric
\(\hat{g}_{\mu\nu} = e^{-2\phi}g_{\mu\nu}\). The dimensionless cosmological
equations in natural units (\(8\pi G/3 = 1\)) are:

\[
H^2 = e^{-4\phi}\rho_m + \frac{1}{2}\dot{\phi}^2 + V(\phi)
\]

\[
\ddot{\phi} + 3H\dot{\phi} + V'(\phi) = e^{-4\phi}\rho_m
\]

\[
\dot{\rho}_m + 3H\rho_m = 3\dot{\phi}\rho_m
\]

\[
V(\phi) = \beta\left(e^{2\phi} - 1 - 2\phi\right)
\]

The model has a single free parameter, the potential scale \(\beta\).
The coupling between the causal field and matter is fixed by the
conformal geometry; no tunable coupling constant appears.

---

## Calibrated Frozen‑Branch Parameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| \(\beta\) | 0.002081 | Potential scale |
| \(\phi_i\) | -0.973092 | Initial causal field |
| \(\dot{\phi}_i\) | 0.360392 | Initial field velocity |

These values are obtained by minimising a cost function that enforces
\(H_0 = 1\), \(\Omega_{m0} = 0.3\), and \(w_0 = -1\) at the present epoch
(\(a = 1\), integrated from \(a_i = 10^{-3}\)). The calibration uses the
Nelder–Mead simplex algorithm.

### Verification Numbers

| Quantity | Value |
| :--- | :--- |
| \(H_0\) | 1.0001 |
| \(\Omega_\phi\) | 0.7000 |
| \(w_0\) | -1.0000 |
| \(\max \left|\Delta H / H_{\Lambda\rm CDM}\right|\) | 0.01% |
| Pantheon+ \(\chi^2\) (276 dof) | 142.70 |

---

## Repository Contents

| File | Purpose |
| :--- | :--- |
| `cft_background.py` | Main script: calibration, integration, Pantheon+ likelihood, CSV export, and figure generation. |
| `validation_tests.py` | Standalone robustness suite: Hamiltonian constraint, attractor behaviour, sensitivity to \(\beta\), and numerical convergence. |
| `README.md` | This file. |
| `LICENSE` | MIT License. |

---

## Requirements

The scripts require only standard scientific Python libraries:

- Python ≥ 3.8
- `numpy`
- `scipy`
- `pandas`
- `matplotlib`

Install the dependencies with:

```bash
pip install numpy scipy pandas matplotlib
