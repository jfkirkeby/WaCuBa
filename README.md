

# WaCuBa -  simple simulation of wave-current-bathymetry interaction
A simple Python module for flexible simulation of linear water waves interacting with currents and variable bathymetry.


<p align="center">
  <img align="middle" width="49%" alt="wave_analysis_final_smaller" src="https://github.com/user-attachments/assets/62465883-cd64-481f-8087-3d1239c17483" />
  <img align="middle" width="49%" alt="wave-current-bathy rough" src="https://github.com/user-attachments/assets/eb6cecdd-3ea7-414a-b87e-9de6c002a58f" />
</p>


## Introduction 

This Notebook contains a short introduction to the WaCuBa module for simulating linear water waves interacting with surface currents and variable bathymetry. Below follows

1. The underlying mathematical model 
2. Code features
3. Numerical method
4. Installation

An introductory Jupyter notebook for setting up and running the module is found [here](notebooks/example1.ipynb). 

More notebooks will be added in the near future. 



### 1. Mathematical model
The underlying system of partial differential equations that are solved numerically is 
```math
\begin{cases}
\partial_t \eta +  \nabla_X\cdot(\bar{U} \eta) = \mathcal{G}(b)\varphi, \\
\partial_t \varphi + \bar{U}\cdot \nabla_{X} \varphi = - g\eta . 
\end{cases}
```

Above, $\eta(t,X)$ is the wave amplitude and $\varphi(t,X)$ is the surface velocity potential. In addition, $\bar{U}(X)$ is the surface current, $b = b(X)$ is the depth (from $z=0$) and $\mathcal{G}(b)$ is the Dirichlet-to-Neumann (DN) operator. The system is described and analysed in detail in [1]. 

### 2. Code features
The code has the following features: 

 - Solves the Cauchy problem the wave system  with prescribed initial conditions $(\eta_0,\varphi_0)$ and variable bathymetry $b(X)$ and current $\bar{U}(X)$ on a rectangular domain. Returns $\eta(t,X),\varphi(t,X)$ and energy density $\mathcal{E}(t,X)$ and additional wave features. 

- Uses ray tracing to compute wavenumber fields $k(X)$ for given bathymetry $b(X)$ and current $\bar{U}(X)$.

- Computes all intrinsic wave properties and solves the asymptotic wave action equation 
```math
\partial_t \mathcal{A} + \nabla_X \left((\bar{U} + C_g)\mathcal{A} )\right) = 0
```
and the Schrödinger equation 
```math
\partial_t A_\varphi + (\bar{U} + C_g)\cdot \nabla_X A_\varphi + \frac{1}{2}\left(\nabla_X \cdot(\bar{U} + C_g)\right) A_\varphi + \frac{D_{\bar{U}+ C_g}\sigma }{2\sigma}A_\varphi  +\frac{i\mu}{2}\nabla_X \cdot (D \nabla_X A_\varphi) = 0.
```

- Allows for flexible and easy comparison of results from different models for a given bathymetry and current.  


### 3. Numerical method

We solve both the wave system and the energy and Schrödinger equations using a standard Fourier pseudo-spectral method (cf. [2]; we express our unknowns in a truncated Fourier basis, e.g., $\eta_N(t,X) = \sum_{|k| \leq N} \eta_{k}(t)e^{i k\cdot X}$, compute spatial derivatives in the $k$-domain, and transform back to physical space for multiplication by vector fields and time stepping. As the current $\bar{U}$ is assumed to be smooth (and there are no non-linear terms), we do not enforce de-aliasing. For the DN operator, we use the truncated Fourier-Galerkin method developed in [3]. We precompute the bathymetry dependent part of the $\mathcal{G}(b)$, and we also implement absorbing boundary conditions [4]. For time integration of the PDEs we use the standard Runge-Kutta 4 scheme and for wavenumber computation, we incorporate the open source ray tracing module [5]. The solver has been verified numerically by considering convergence as a function of grid size/Fourier modes, and by veryfying that the total energy is conserved in the case of variable bathymetry and divergence free currents. 

### 4. Installation

To run WaCuBa on locally, clone this repository and install the necessary dependencies using `pip`. 

Open your terminal and run the following commands:

```bash
# Clone the repository
git clone [https://github.com/jfkirkeby/WaCuBa.git](https://github.com/jfkirkeby/WaCuBa.git)

# Navigate into the project folder
cd WaCuBa

# Install the required libraries
pip install -r requirements.txt

```

### References: 

[1] A. Kirkeby & T. Halsne - Wave-Current-Bathymetry Interaction Revisited: Modeling, Analysis and Asymptotics, 2026,  [Pre-print](https://arxiv.org/abs/2603.25435)

[2] N. Trefethen, Spectral Methods in MATLAB, 2000, SIAM.

[3] D. Andrade & A. Nachbin, A three-dimensional Dirichlet-to-Neumann operator for water waves over topography, 2018, Journal of Fluid Mechanics.

[4] D. Bodony, Analysis of sponge zones for computational fluid mechanics, 2006, Journal
of Computational Physics.

[5] T. Halsne et. al., Ocean wave tracing v. 1: a numerical solver of the wave ray equations for ocean waves on variable currents at arbitrary depths, 2023, Geoscientific Model Development.
