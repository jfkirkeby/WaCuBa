# WaCuBa
A simple Python module for flexible simulation of linear water waves interacting with currents and variable bathymetry. Code coming soon.

## Introduction 

This Notebook contains a short introduction to the WaCuBa module for simulating linear water waves interacting with surface currents and variable bathymetry. 

1. The underlying mathematical model 
2. Code features
3. Numerical method
4. Discretization, parameter tuning and limitations
5. Worked examples



### 1. Mathematical model
The underlying system of partial differential equations that are solved numerically is 
```math
\begin{cases}
\partial_t \eta +  \nabla_X\cdot(\bar{U} \eta) = \mathcal{G}(b)\varphi, \\
\partial_t \varphi + \bar{U}\cdot \nabla_{X} \varphi = - g\eta . 
\end{cases}
```

Above, $\eta(t,X)$ is the wave amplitude and $\varphi(t,X)$ is the surface velocity potential. In addition, $\bar{U}(X)$ is the surface current, $b = b(X)$ is the depth (from $z=0$) and $\mathcal{G}(b)$ is the Dirichlet-to-Neumann (DN) operator. The system is described and analysed in detail in the paper (....). 

### 2. Code features
The code has the following features: 

 - Solves the Cauchy problem the wave system  with prescribed initial conditions $(\eta_0,\varphi_0)$ and variable bathymetry $b(X)$ and current $\bar{U}(X)$ on a rectangular domain. Returns $\eta(t,X),\varphi(t,X)$ and energy density $\mathcal{E}(t,X)$ and additional wave features. 

- Uses ray tracing to compute wavenumber fields ```math \bm{k}(X) ``` for given bathymetry $b(X)$ and current $\bar{U}(X)$.

- Computes all intrinsic wave properties and solves the asymptotic wave action equation 
$$ \partial_t \mathcal{A} + \nabla_X \left((\bar{U} + C_g)\mathcal{A} )\right) = 0$$
and the Schrödinger equation 
$$ \partial_t A_\varphi + (\bar{U} + C_g)\cdot \nabla_X A_\varphi + \frac{1}{2}\left(\nabla_X \cdot(\bar{U} + C_g)\right) A_\varphi + \frac{D_{\bar{U}+ C_g}\sigma }{2\sigma}A_\varphi  +\frac{i\mu}{2}\nabla_X \cdot (D \nabla_X A_\varphi) = 0. $$

- Allows for flexible and easy comparison of results from different models for a given bathymetry and current.  


### 3. Numerical method

We solve both the wave system and the energy and Schrödinger equations using a standard Fourier pseudo-spectral method (cf. \cite{trefethen2000spectral}); we express our unknowns in a truncated Fourier basis, e.g., $\eta_N(t,X) = \sum_{|\bm{k}| \leq N} \eta_{\bm{k}}(t)e^{i\bm{k}\cdot X}$, compute spatial derivatives in the $\bm{k}$-domain, and transform back to physical space for multiplication by vector fields and time stepping. As the current $\bar{U}$ is assumed to be smooth (and there are no non-linear terms), we do not enforce de-aliasing. For the DN operator, we use the truncated Fourier-Galerkin method developed in \cite{andrade2018three}. We precompute the bathymetry dependent part of the $\mathcal{G}(b)$, and we also implement absorbing boundary conditions \cite{bodony2006analysis}. For time integration of the PDEs we use the standard Runge-Kutta 4 scheme and for wavenumber computation, we incorporate the open source ray tracing module \cite{halsne2023ocean}. The solver has been verified numerically by considering convergence as a function of grid size/Fourier modes, and by veryfying that the total energy is conserved in the case of variable bathymetry and divergence free currents. 

### 4. Discretization, parameter tuning and limitations

In the example below we outline some simple heuristics for choosing the spatial and temporal discretization, for tuning the absorbing boundary layer and for pre-computing the DN operator to a sufficient accuracy. 
