# NV center project

This repository contains the code I developed for the project on NV centers in collaboration with Dominik.

## Zeeman splitting

The sign of the gyromagnetic ratio determines if the spin is parallel (positive) or antiparallel (negative) to its magnetic moment and the magnetic field.

$$
    H = -\mu B = -\gamma \hbar S B.
$$

## Magnetic Dipole Interaction

The dipolar coupling Hamiltonian describes the interaction between two magnetic dipoles, e.g., between two spins.

$$
    H_{\text{dip}} = -\frac{\mu_0}{4\pi} \frac{\gamma_e \gamma_C}{r^3} \hbar^2 \left( 3 ( S_1 \cdot n ) ( S_2 \cdot n ) - S_1 \cdot S_2 \right).
$$

This yields the angular frequency $\omega_{\text{dip}} = H_{\text{dip}}/\hbar$ and the frequency $f_{\text{dip}} =  H_{\text{dip}}/h$. 

## Cluster Expansion

Given and a matrix $S$ accounting for the pulse sequence, the state $\rho(0)$ evolves as follows:

$$
    \rho(t) = \frac{1}{M} \sum_{n=1}^M \rho_{\mathrm{E} 0}^{(n)}(t) + \rho_{\mathrm{E} 1}^{(n)}(t) + \rho_{\mathrm{E} 2}^{(n)}(t) + ...
$$

where 

$$
    \rho_{\mathrm{E} 0}^{(n)}\left(t\right)=\mathcal{S}_{n} \tilde{\rho}_{\mathrm{E} 0}^{(n)}(0) \mathcal{S}_{n}^{\dagger}\\
    \rho_{\mathrm{E} 1}^{(n, l)}\left(t\right)=\operatorname{tr}_{B, l}\left(\mathcal{S}_{n, l} \tilde{\rho}_{\mathrm{E} 1}^{(n, l)}(0) \mathcal{S}_{n, l}^{\dagger}\right)\\
    \rho_{\mathrm{E} 2}^{(n, l, q)}\left(t\right)=\operatorname{tr}_{B, l, q}\left(\mathcal{S}_{n, l, q} \tilde{\rho}_{\mathrm{E} 2}^{(n, l, q)}(0) \mathcal{S}_{n, l, q}^{\dagger}\right)
$$

and 

$$
    \langle a| \rho_{\mathrm{E} 1}^{(n)}(t)|b\rangle   = \langle a| \rho_{\mathrm{E} 0}^{(n)}(t)|b\rangle \prod_l \frac{\langle a| \rho_{\mathrm{E} 1}^{(n, l)}(t)|b\rangle}{\langle a| \rho_{\mathrm{E} 0}^{(n)}(t)|b\rangle}\\
    \langle a| \rho_{\mathrm{E} 2}^{(n)}(t)|b\rangle = \langle a| \rho_{\mathrm{E} 1}^{(n)}(t)|b\rangle \prod_{l, q} \frac{\langle a| \rho_{\mathrm{E} 2, q)}^{(n, l)}(t)|b\rangle}{\langle a| \rho_{\mathrm{E} 0}^{(n)}(t)|b\rangle^{-1}\langle a| \rho_{\mathrm{E} 1}^{(n)}(t)|b\rangle\langle a| \rho_{\mathrm{E} 1}^{(n, q)}(t)|b\rangle}
$$

To avoid numerical errors, the cluster expansion can directly be applied to the fidelities:

$$
\mathcal{F}_{f, \mathrm{E} 1}^{(n)}=\mathcal{F}_{f, \mathrm{E} 0}^{(n)} \prod_l \frac{\mathcal{F}_{f, \mathrm{E} 1}^{(n, l)}}{\mathcal{F}_{f, \mathrm{E} 0}^{(n)}} 
$$

$$
\mathcal{F}_{f, \mathrm{E} 2}^{(n)}= \mathcal{F}_{f, \mathrm{E} 1}^{(n)} \prod_{l, q} \frac{\mathcal{F}_{f, \mathrm{E} 2}^{(n, l, q)}}{\left(\mathcal{F}_{f, \mathrm{E} 0}^{(n)}\right)^{-1} \mathcal{F}_{f, \mathrm{E} 1}^{(n, l)} \mathcal{F}_{f, \mathrm{E} 1}^{(n, q)}}.
$$