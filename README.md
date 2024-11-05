# NV center project

This repository contains the code I developed for the project on NV centers in collaboration with Dominik.

## Theory

### Zeeman splitting

The sign of the gyromagnetic ratio determines if the spin is parallel (positive) or antiparallel (negative) to its magnetic moment. Since the magnetic moment couples to the magnetic field and gives lowest energy if they are parallel and highest energy if they are antiparallel,

$$
    H = -\mu B = -\gamma \hbar m_s B.
$$

Be careful:
If we set $\hbar=1$, the Hamiltonian is given in the unit of a angular frequency (rad/s) and we have to use the gyromagnetic ratio in rad/(Ts). If we set $h=1$, the Hamiltonian is given in the unit of a frequency (1/s) and we have to use the reduced gyromagnetic ratio in 1/(Ts).

### Dipolar coupling

The dipolar coupling describes the direct interaction between two magnetic dipoles.

$$
    H_{\text{dip}} = \frac{1}{r^3} \cdot \frac{\mu_0}{4 \pi} \cdot g_1 g_2 \mu_{\mathrm{B}}^2 \left( \vec{S}_1 \cdot \vec{S}_2 - \frac{3}{r^2} \left( \vec{S}_1 \cdot \vec{r} \right) \left( \vec{S}_2 \cdot \vec{r} \right) \right) \\
    = -\frac{\hbar^2 \mu_0}{4\pi} \frac{\gamma_e \gamma_C}{r^3} \left( 3 \left( \vec{S}_1 \cdot \vec{n} \right) \left( \vec{S}_2 \cdot \vec{n} \right) - \vec{S}_1 \cdot \vec{S}_2 \right)
$$
