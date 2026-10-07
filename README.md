```markdown
# Causal Field Theory: Cosmological Background and Observational Confrontation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey.svg)]()

A reproducible numerical pipeline for the cosmological sector of **Causal Field Theory (CFT)** — a self-contained extension of General Relativity rooted in the *Principle of Causal Optimality*. This repository provides the complete code and data used in the accompanying manuscript to obtain the frozen-branch solution, verify its internal consistency, and confront its predictions with the Pantheon+ compilation of Type Ia supernovae.

---

## Table of Contents

1. [Scientific Background](#scientific-background)
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Output Files](#output-files)
6. [Physical Framework](#physical-framework)
7. [Numerical Methods](#numerical-methods)
8. [Key Results](#key-results)
9. [The Frame Hierarchy and the Number 18](#the-frame-hierarchy-and-the-number-18)
10. [Verification of Results](#verification-of-results)
11. [Testing](#testing)
12. [Reproducibility](#reproducibility)
13. [Citation](#citation)
14. [License](#license)
15. [Contact](#contact)

---

## Scientific Background

Causal Field Theory (CFT) proposes that the causal structure of spacetime is not a fixed arena but a dynamical entity, whose configuration is selected by a single variational principle — the **Principle of Causal Optimality**:

$$
\delta \int \frac{1}{\acute{c}^{2}} \, d\tau = 0,
$$

where $\acute{c}$ is the *effective causal speed* and $d\tau$ is the proper time. The effective causal speed is defined through the **causal Lorentz factor** $\gamma' \equiv 1/\sqrt{1 - v^{2}/\acute{c}^{2}}$, which emerges as the unique solution of a Lorentz-type self-consistency condition, yielding the quadratic relation $\acute{c}^{2} = c^{2} + v^{2}$. This definition, together with the principle, defines a dimensionless scalar field $\phi$ via $\acute{c} = c\, e^{\phi}$, with the key identification $\phi = \ln\gamma'$.

The theory is formulated in the **Einstein frame**, where matter fields couple to the physical metric

$$
\hat{g}_{\mu\nu} = e^{-2\phi}\, g_{\mu\nu},
$$

and the total action combines the Einstein–Hilbert term, a canonical kinetic term, and the minimal potential consistent with the physical requirements of vacuum stability, positivity, and exponential saturation:

$$
V(\phi) = \beta \left( e^{2\phi} - 1 - 2\phi \right).
$$

A central element of this implementation is the proper treatment of the **frame transformation** between the Einstein and physical (Jordan) frames, which is essential for a correct confrontation with observational data. Specifically, the code distinguishes between the Einstein-frame Hubble parameter $H_E(z_E)$ and the physical-frame Hubble parameter $H_{\rm phys}(z_{\rm phys})$, and uses the latter to compute the luminosity distance that is compared with supernova observations.

The present code implements the **cosmological sector** of CFT, focusing on the *physical frozen-branch solution* that provides a dynamical origin for dark energy.

---

## Repository Structure

```

.
├── cft_background.py                 # Main integration and figure generation
├── validation_tests.py               # Independent robustness checks (App. A.4)
├── README.md                         # This file
├── LICENSE                           # MIT License
│
├── background_physical.csv           # Background evolution (output)
├── pantheon_comparison_physical.csv  # SNe Ia comparison (output)
├── verification_numbers_physical.csv # Key cosmological quantities (output)
│
├── fig_background_evolution.pdf      # Publication-ready figure (PDF)
└── fig_background_evolution.png      # Publication-ready figure (PNG)

```

---

## Installation

### Prerequisites

The code is self-contained and requires only standard scientific Python libraries:

```bash
pip install numpy scipy pandas matplotlib
```

Recommended versions

Package Minimum version
Python 3.8
numpy 1.20
scipy 1.7
pandas 1.3
matplotlib 3.4

Optional (for development):

· pytest ≥ 6.0
· black ≥ 21.0

Clone the Repository

```bash
git clone https://github.com/MahmoudFAbdelSattar/cft_cosmology.git
cd cft_cosmology
```

---

Usage

Main Analysis

To run the full pipeline — including the integration of the background equations, the computation of the $\chi^{2}$ statistic against Pantheon+, the internal consistency checks, the export of CSV data files, and the generation of the six-panel figure — execute:

```bash
python cft_background.py
```

The script will print a detailed log of its progress and save all output files to the current working directory.

Validation Suite

To run the independent robustness checks described in Appendix A.4 of the manuscript:

```bash
python validation_tests.py
```

This script performs the following tests without re-calibrating the model:

1. Hamiltonian constraint — verifies that the Friedmann equation is satisfied to machine precision.
2. Sensitivity to $\beta$ — quantifies the response of the present-day observables to a $\pm 5\%$ variation of the potential scale.
3. Numerical convergence — establishes that the solution is stable under tightened ODE tolerances.

Note. The validation_tests.py script attempts to import from cft_background.py. If the companion module is not on the Python path (for example, when the script is executed inside a Google Colab cell without uploading the companion file), the script falls back to self-contained local definitions. The numerical results are identical in both modes; only the log message differs (importing from cft_background.py versus running in standalone mode).

Google Colab

Both scripts are fully self-contained and may be executed directly in a Google Colab cell. No additional setup is required.

Quick start:

1. Upload cft_background.py and validation_tests.py to Colab (or paste their contents into separate cells).
2. Run:

```python
!python cft_background.py
!python validation_tests.py
```

Alternatively, copy the entire contents of each script into a Colab cell and execute the cell directly.

---

Output Files

Background Evolution — background_physical.csv

A tabulated record of the cosmological solution on the integration grid:

Column Description
z_E Einstein-frame redshift
z_phys Physical-frame redshift ($z_{\rm obs}$)
a_E Einstein-frame scale factor
phi Causal field $\phi$
v Field velocity $\dot{\phi}$
rho_m Matter energy density
H_E Einstein-frame Hubble parameter
H_phys Physical-frame Hubble parameter
Omega_m Matter density parameter
Omega_phi Dark-energy density parameter
w_phi Equation of state

Supernova Comparison — pantheon_comparison_physical.csv

Per-supernova residuals against the CFT prediction:

Column Description
z Observed redshift
mu_obs Observed distance modulus
sigma_mu Measurement uncertainty
mu_CFT CFT prediction
residual $\mu_{\rm obs} - \mu_{\rm CFT}$

Verification Numbers — verification_numbers_physical.csv

A single-row summary of the key cosmological quantities at the present epoch, suitable for direct comparison with the manuscript. Includes:

Field Description
phi_0 Present-day causal field value
w_0 Present-day equation of state
H_E_today Einstein-frame Hubble parameter
H_phys_today Physical-frame Hubble parameter
Omega_phi Dark-energy density parameter
V_phi_0 Potential energy today
gamma_prime_0 Causal Lorentz factor $\gamma'_0 = e^{\phi_0}$
v0_over_HE Present-day field velocity $\dot{\phi}_0/H_E$
H_phys_over_H_E Frame hierarchy ratio $H_{\rm phys}/H_E$
chi2 $\chi^{2}$ statistic
dof Degrees of freedom
chi2_dof Reduced $\chi^{2}$
M_best Marginalised magnitude offset
sigma_res RMS of residuals

---

Physical Framework

Frame Transformation

The theory is solved in the Einstein frame but compared with observations in the physical (Jordan) frame. The transformation rules are:

· Redshift:
  1 + z_{\rm phys} = e^{\phi - \phi_0} \left( 1 + z_E \right)
· Hubble parameter:
  H_{\rm phys} = e^{\phi} \left( H_E - \dot{\phi} \right)
· Luminosity distance:
  D_L^{\rm phys}(z) = (1+z) \int_0^z \frac{dz'}{H_{\rm phys}(z')}

A key point of the framework is that the observed redshift satisfies $z_{\rm obs} = z_{\rm phys}$, because the electron mass is constant in the physical frame.

Equations of Motion

The Einstein-frame background system is:

\begin{aligned}
H_E^{2} &= e^{-4\phi}\,\rho_m + \tfrac{1}{2}\,\dot{\phi}^{2} + V(\phi), \\
\ddot{\phi} + 3H_E\dot{\phi} + V'(\phi) &= e^{-4\phi}\,\rho_m, \\
\dot{\rho}_m + 3H_E\rho_m &= 3\dot{\phi}\,\rho_m.
\end{aligned}

---

Numerical Methods

· ODE solver: scipy.integrate.solve_ivp with the explicit Runge–Kutta 5(4) method (RK45).
· Relative tolerance: $10^{-10}$.
· Absolute tolerance: $10^{-13}$.
· Maximum step size: $10^{-3}$ (in Hubble-time units).
· Integration range: from $a_i = 10^{-3}$ ($z_i \approx 1000$) to $a = 1$.
· Termination: an event function halts the integration precisely when $a = 1$.
· Quadrature for $D_L^{\rm phys}$: cumulative trapezoidal rule on the sorted $(z_{\rm phys}, H_{\rm phys})$ grid.
· Calibration: Nelder–Mead simplex algorithm (scipy.optimize.minimize), minimising a weighted cost function of $H_E(z=0)$, $\Omega_{m0}$, and $w_0$.

---

Key Results

The calibrated frozen-branch solution yields:

Quantity Value
Potential scale $\beta$ $0.002081$
Initial field $\phi_i$ $-0.973092$
Initial velocity $\dot{\phi}_i$ $0.360392$
Present field $\phi_0$ $2.919250$
$H_E(z=0)$ $1.000057$
$H_{\rm phys}(z=0)$ $18.582824$
$\Omega_\phi$ (Einstein frame) $0.700026$
$w_0$ $-0.999988$
$\chi^{2}$ (Pantheon+) $138.28$
$\chi^{2}/{\rm dof}$ $0.501$
$\sigma_{\rm residuals}$ $0.142$ mag

The Friedmann constraint closure, $\Omega_m(z) + \Omega_\phi(z) = 1$, is satisfied to machine precision (maximum relative deviation $4.44 \times 10^{-16}$). The Hamiltonian constraint $H_E^{2} = e^{-4\phi}\rho_m + \tfrac{1}{2}\dot{\phi}^{2} + V(\phi)$ is likewise satisfied to machine precision (maximum relative deviation $2.22 \times 10^{-16}$). These checks confirm the numerical robustness of the solution.

---

The Frame Hierarchy and the Number 18

The present-day ratio between the physical and Einstein-frame Hubble parameters is

\frac{H_{\rm phys}(z=0)}{H_E(z=0)} = e^{\phi_0}\left(1 - \frac{\dot{\phi}_0}{H_E}\right) \approx 18.58.

This numerical hierarchy deserves careful interpretation, as it is easily misread. The pipeline prints the following diagnostic quantities:

Quantity Value Interpretation
$\dot{\phi}_0/H_E$ $-2.935409 \times 10^{-3}$ Present-day field velocity (residual)
$e^{\phi_0} = \gamma'_0$ $18.527384$ Causal Lorentz factor today
$H_{\rm phys}/H_E$ $18.581769$ Frame hierarchy ratio
$d\hat{t}/dt_E\|_{z=0}$ $5.397416 \times 10^{-2}$ Time-scale ratio between frames

Mathematical origin

The hierarchy is not a free parameter. At saturation ($w_0 = -1$), the exact relation is

e^{2\phi_0} = \frac{\Omega_\phi H_E^{2} - \frac{1}{2}\dot{\phi}_0^{2}}{\beta} + 1 + 2\phi_0,

where the kinetic term $\frac{1}{2}\dot{\phi}_0^{2} \approx 4.5 \times 10^{-6}$ is negligible. Substituting the calibrated parameters gives $e^{\phi_0} \approx 18.53$, in excellent agreement with the numerical value $18.58$. The small difference ($\sim 0.3\%$) arises from the residual field velocity $\dot{\phi}_0/H_E \approx -2.9 \times 10^{-3}$.

Physical meaning

The ratio reflects the different time scales measured in the two frames. From the conformal relation $\hat{g}_{\mu\nu} = e^{-2\phi}g_{\mu\nu}$, the physical and Einstein-frame time intervals are related by $d\hat{t} = e^{-\phi}\,dt_E$, so at the present epoch $d\hat{t}/dt_E \approx 0.054$. This is a conformal transformation between two descriptions of the same physics, not a physical effect within a single frame.

The ratio does not imply that the Universe expands $18.58$ times faster in any absolute sense; the Einstein and physical frames describe the same physical expansion in different units. The ratio is a consequence of the calibrated model, not a fundamental constant of nature.

Relation to the age of the Universe

The ratio $18.58$ is a local quantity, evaluated at $z = 0$. It does not directly determine the age of the Universe, which is an integral over the entire cosmic history:

\hat{t}_0 = \int_0^{t_0} e^{-\phi(t)}\,dt_E.

The integral receives significant contributions from both the early-time regime (where $e^{-\phi} > 1$ amplifies the integrand) and the late-time regime (where the time interval $dt_E$ is larger but $e^{-\phi} \ll 1$ suppresses the integrand). The resulting physical age is therefore not simply related to the present-day ratio $18.58$.

---

Verification of Results

After running cft_background.py, the following values should match those reported in the manuscript:

Quantity Expected value
$\chi^{2}$ (Pantheon+) $138.28$
$\chi^{2}/{\rm dof}$ $0.501$
$w_0$ $-0.999988$
$\Omega_\phi$ $0.700026$
$H_{\rm phys}(z=0)$ $18.582824$
$\dot{\phi}_0/H_E$ $-2.935409 \times 10^{-3}$
$e^{\phi_0} = \gamma'_0$ $18.527384$
$H_{\rm phys}/H_E$ $18.581769$
$d\hat{t}/dt_E\|_{z=0}$ $5.397416 \times 10^{-2}$

If any value deviates, verify:

1. The Pantheon+ data file is complete and unmodified.
2. The Python package versions satisfy the minimum requirements.
3. The ODE tolerances are not relaxed.

---

Testing

To verify the installation and the numerical pipeline, run:

```bash
python validation_tests.py
```

Expected output:

· Test 1 (Hamiltonian constraint): maximum relative deviation $< 3 \times 10^{-16}$ ✓
· Test 2 ($\beta$ sensitivity): spread in $w_0$ $< 3 \times 10^{-4}$ ✓
· Test 3 (numerical convergence): max $|\Delta H_E|$ $< 3 \times 10^{-11}$ ✓

If all three tests pass, the installation is correct.

---

Reproducibility

All results presented in the accompanying manuscript are fully reproducible by executing the two Python scripts in the order described above. The pipeline is deterministic and does not rely on external data beyond the publicly available Pantheon+ compilation.

Note on sensitivity. The frozen-branch solution is the unique trajectory that simultaneously satisfies the three calibration conditions ($H_E(z=0)=1$, $\Omega_{m0}=0.3$, $w_0=-1$). Small variations in the initial conditions $(\phi_i, \dot{\phi}_i)$ shift the present-day observables, as expected for freezing scalar-field cosmologies with a shallow exponential potential. This sensitivity does not affect reproducibility: the calibrated parameters are fixed, and the pipeline is deterministic once they are specified.

The Pantheon+ Hubble-flow sample is retrieved directly from the public repository:

```
https://github.com/PantheonPlusSH0ES/DataRelease
```

---

Citation

If you use this code or the results presented here in your own work, please cite the accompanying manuscript:

```bibtex
@article{AbdelSattar2026CFT,
  author  = {Abdel-Sattar, Mahmoud F.},
  title   = {Causal Field Theory: Dynamical Dark Energy from the
             Principle of Causal Optimality},
  journal = {Foundations of Physics},
  year    = {2026},
  note    = {Submitted},
  doi     = {pending}
}
```

---

License

This project is licensed under the MIT License — see the LICENSE file for details.

---

Contact

Mahmoud F. Abdel-Sattar
Department of Astronomy and Meteorology
Faculty of Science, Al-Azhar University
Cairo, Egypt

📧 m.f.abdel-sattar@azhar.edu.eg

---

Acknowledgments

The author thanks the Pantheon+ collaboration for making their supernova data publicly available. This work was carried out independently and received no external funding.

```
