import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from numpy.fft import fft2, ifft2, fftfreq
from numpy.linalg import cond, solve
from dataclasses import dataclass
import scipy as sp  
import scipy.optimize, scipy.integrate
import xarray as xa
import sys,os




# ============================================================
# 1. Core numerics: grid, wavenumbers, sponge, upwind
# ============================================================

def build_grid(Lx, Ly, Nx, Ny):
    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    dx = Lx / Nx
    dy = Ly / Ny
    X, Y = np.meshgrid(x, y, indexing="xy")
    return X, Y, dx, dy

def build_wavenumbers(Lx, Ly, Nx, Ny):
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=Lx/Nx)
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=Ly/Ny)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    K = np.sqrt(KX**2 + KY**2)
    return KX, KY, K



def build_sponge_sides(Lx, Ly, X, Y, sponge_frac=0.15, sigma_max=1.0, sides="lrtb"):
    """
    Builds a sponge layer for the absorbing boundary condition. 
    Lx = domain length in x
    Ly = domain length in y
    X, Y = meshgrid of coordinates
    sponge_frac = fraction of domain to use as sponge layer (e.g. 0.1 for 10% of domain)
    sigma_max = damping coefficient
    sides: "lr" for left/right only, "tb" for top/bottom only, "lrtb" for all.
    """
    distances = []

    if "l" in sides:
        distances.append(X)
    if "r" in sides:
        distances.append(Lx - X)
    if "b" in sides:
        distances.append(Y)
    if "t" in sides:
        distances.append(Ly - Y)

    # Calculate the distance to the nearest active boundary
    dist_min = np.minimum.reduce(distances)
    if "l" in sides and "r" in sides:
        width = sponge_frac * min(Lx, Ly)
    elif "t" in sides and "b" in sides:
        width = sponge_frac * min(Lx, Ly)
    
        
    sigma = np.zeros_like(X)
    
    mask = dist_min < width
    sigma[mask] = sigma_max * ((width - dist_min[mask]) / width)**3
    
    return sigma

def spectral_gradient(f, KX, KY):
    """
    Computes gradients using FFT.
    KX, KY are meshgrids of wavenumbers.
    """
    f_hat = fft2(f)
    
    # Derivative in Fourier space is multiplication by i*k
    dfdx_hat = 1j * KX * f_hat
    dfdy_hat = 1j * KY * f_hat
    
    dfdx = np.real(ifft2(dfdx_hat))
    dfdy = np.real(ifft2(dfdy_hat))
    
    return dfdx, dfdy

# ============================================================
# 2. DtN Operator (Andrade & Nachbin 2018)
# ============================================================

def precompute_dtn_system(Lx, Ly, Nx, Ny, b, k_cutoff=0.2, flat_bottom=False):
    """
    Computes the DtN operator matrices for water waves over variable topography 
    using the Fourier-Galerkin method described in Andrade & Nachbin (2018).
    
    Args:
    
        Lx = domain length in x
        Ly = domain length in y
        X, Y = meshgrid of coordinates
        b = depth profile (2D array of shape (Ny, Nx)) (b > 0)
        k_cutoff = maximum wavenumber to include in Galerkin expansion 
        flat_bottom = if True, only compute flat bottom contribution (no variable topography)
    
    Returns: dict with:
    
        'T_matrix': matrix mapping surface Fourier coeffs to topographic coeffs
        'active_indices': indices of active modes in Fourier space
        'flat_factor': array of shape (Ny, Nx) for flat bottom contribution in Fourier space
        'topo_recon_factor': array of shape (n_active,) for reconstructing topo contribution
        'flat_bottom': boolean flag for flat bottom case
        'shape': original grid shape (Ny, Nx) for reference
    """
    

    if b.shape != (Ny, Nx):
        # Try to fix transposed input automatically
        if b.shape == (Nx, Ny):
            print(f"Notice: Transposing b from {b.shape} to {(Ny, Nx)} to match simulation grid.")
            b = b.T
        else:
            raise ValueError(f"b shape {b.shape} must match (Ny, Nx)={(Ny, Nx)}")

    
   
    KX, KY, K_mod = build_wavenumbers(Lx, Ly, Nx, Ny)  # Get KX, KY with correct indexing
 
    K_mod[0, 0] = 1.0  

    # 3. Depth Decomposition
    h = np.mean(b)
    H = b - h
   
    # 4. Galerkin Mode Selection
    
    # Mask active modes (Low frequency & exclude mean)
    mask_active = (K_mod < k_cutoff)
    mask_active[0, 0] = False 
    
    active_indices = np.where(mask_active)
    # active_indices[0] are rows (y), active_indices[1] are cols (x)
    n_active = len(active_indices[0])
    
    if flat_bottom:
        flat_factor = K_mod * np.tanh( K_mod * h)
        flat_factor[0, 0] = 0.0 # Zero mean
        T_matrix = np.zeros((1, 1))
        topo_recon_factor = np.zeros(0)
    
    else:
    
        print(f"System Setup: {Nx}x{Ny} grid. Active modes: {n_active} "
            f"({100*n_active/(Nx*Ny):.1f}%). Mean depth h={h:.2f}.")

        if n_active == 0:
            T_matrix = np.zeros((1, 1))
            # Placeholders
            kx_active = np.array([])
            ky_active = np.array([])
            k_active = np.array([])
        else:
            # Extract active wavenumbers
            kx_active = KX[active_indices]
            ky_active = KY[active_indices]
            k_active = K_mod[active_indices]
            
            # 5. Build Matrices A and B
            mat_A = np.zeros((n_active, n_active), dtype=np.complex128)
            mat_B = np.zeros((n_active, n_active), dtype=np.complex128)
            
            # Loop over active modes (columns j)
            for j in range(n_active):
                k_j = k_active[j]
                k_x_j = kx_active[j]
                k_y_j = ky_active[j]
                
                # Spatial terms
                val_A = np.sinh(k_j * H) / np.cosh(k_j * h)
                val_B = np.cosh(k_j * (h + H)) / (np.cosh(k_j * h)**2)
                
                fft_A = fft2(val_A)
                fft_B = fft2(val_B)
                
                # Calculate shift indices (l - k) for convolution
                # active_indices[0] is y-index (rows), active_indices[1] is x-index (cols)
                l_y = active_indices[0]
                l_x = active_indices[1]
                
                # Target k indices for this column
                k_y_idx = active_indices[0][j]
                k_x_idx = active_indices[1][j]
                
                shift_y = (l_y - k_y_idx) % Ny
                shift_x = (l_x - k_x_idx) % Nx
                
                coeffs_A = fft_A[shift_y, shift_x]
                coeffs_B = fft_B[shift_y, shift_x]
                
                # Dot product (l . k)
                l_dot_k = kx_active * k_x_j + ky_active * k_y_j
                
                # Fill column j
                mat_A[:, j] = 1j * (l_dot_k / k_j) * coeffs_A
                mat_B[:, j] = 1j * (l_dot_k / (k_j**2)) * coeffs_B

            # 6. Solve T = B_inv * A
            c_num = cond(mat_B)
            print(f"Matrix Condition Number: {c_num:.2e}")
            if c_num > 1e12:
                print("WARNING: Matrix ill-conditioned. Reduce k_cutoff.")
                
            T_matrix = solve(mat_B, mat_A)
            topo_recon_factor = np.zeros(0)
            
            if n_active > 0:
                topo_recon_factor = 1.0/ (np.cosh(k_active * h)**2)


        # 7. Reconstruction Factors
        flat_factor =  K_mod * np.tanh(K_mod * h)
        flat_factor[0, 0] = 0.0 

    return {
        'T_matrix': T_matrix,
        'active_indices': active_indices,
        'flat_factor': flat_factor,
        'topo_recon_factor': topo_recon_factor,
        'flat_bottom': flat_bottom,
        'shape': (Ny, Nx) # Store as (rows, cols)
    }

def apply_dtn(phi, topo_data):
    """
    Applies the DtN operator G[q] to surface potential q.
    
    Args: 
        
        phi = surface potential (2D array of shape (Ny, Nx))
        topo_data = dict returned by precompute_dtn_system 
        
    Returns: action of DN operator on phi, array of shape (Ny, Nx)
    
    """
    # 1. FFT
    phi_hat = fft2(phi)
    
    # 2. Flat Bottom Contribution
    g_hat = phi_hat * topo_data['flat_factor']
    
    if topo_data['flat_bottom'] == False:
        # 3. Variable Bottom Contribution
        idx = topo_data['active_indices']
        if len(idx[0]) > 0:
            phi_active = phi_hat[idx]
            
            # Map surface coeffs to topographic coeffs: X = T * q
            X_coeffs = np.dot(topo_data['T_matrix'], phi_active)
            
            # Reconstruct and add to g_hat
            g_hat[idx] += X_coeffs * topo_data['topo_recon_factor']
        
    # 4. Inverse FFT
    return np.real(ifft2(g_hat))




# ============================================================
# 3. Time integrator for wave system
# ============================================================

def rhs_system(eta, phi, Ux, Uy, divU, g, Kx, Ky, sigma, topo_data, S=0.0):

    # Spectral gradients 
    d_eta_dx, d_eta_dy = spectral_gradient(eta, Kx, Ky)
    d_phi_dx, d_phi_dy = spectral_gradient(phi, Kx, Ky)
    
    # Advection terms
    adv_eta = Ux * d_eta_dx + Uy * d_eta_dy
    adv_phi = Ux * d_phi_dx + Uy * d_phi_dy
    
    # Vertical velocity from DtN
    Gphi = apply_dtn(phi, topo_data)

    # Evolution equations with sponge layers
    eta_t = -adv_eta -divU*eta + Gphi - sigma * eta + S
    phi_t = -adv_phi - g * eta - sigma * phi

    return eta_t, phi_t

def step_system_rk4(eta, phi, dt, Ux, Uy,divU, g, Kx, Ky, sigma, topo_data, source=None, t=0.0):
    
    """ One RK4 step from t to t + dt. source is an optional callable S(t) added to the eta equation. """
    
    if source is None:
        S1 = S2 = S4 = 0.0
    else:
        S1, S2, S4 = source(t), source(t + 0.5*dt), source(t + dt)
    
    k1_eta, k1_phi = rhs_system(eta, phi, Ux, Uy,divU, g, Kx, Ky, sigma, topo_data, S1)
    
    k2_eta, k2_phi = rhs_system(
        eta + 0.5*dt*k1_eta,
        phi + 0.5*dt*k1_phi,
        Ux, Uy, divU, g, Kx, Ky, sigma, topo_data, S2
    )
    
    k3_eta, k3_phi = rhs_system(
        eta + 0.5*dt*k2_eta,
        phi + 0.5*dt*k2_phi,
        Ux, Uy, divU, g, Kx, Ky, sigma, topo_data, S2
    )
    
    k4_eta, k4_phi = rhs_system(
        eta + dt*k3_eta,
        phi + dt*k3_phi,
        Ux, Uy, divU, g, Kx, Ky, sigma, topo_data, S4
    )

    eta_new = eta + (dt/6.0) * (k1_eta + 2*k2_eta + 2*k3_eta + k4_eta)
    phi_new = phi + (dt/6.0) * (k1_phi + 2*k2_phi + 2*k3_phi + k4_phi)
    return eta_new, phi_new





def simulate_wave_system(
    Lx, Ly, Nx, Ny,
    T_final, dt,
    eta0, phi0,
    U,
    topo_data,
    sponge_frac=0.15,
    sponge_sides="lrtb", 
    sigma_max=1.0,
    snapshot_interval=None,
    source=None,
    callback=None,
    ):
    
    """ 
    Simulates the wave system using RK4 time stepping and spectral methods for spatial derivatives.
    
    Args: 
        Lx, Ly, Nx, Ny: domain and resolution parameters
        T_final, dt: simulation time and temporal stepsize
        eta0, phi0: initial conditions for surface elevation and potential (2D arrays)
        U: tuple (Ux, Uy) of current velocity fields (2D arrays)
        topo_data: precomputed DtN operator data from precompute_dtn_system
        sponge_frac: fraction of domain to use as sponge layer for absorbing boundaries
        sponge_sides: which sides to apply sponge layer ("l", "r", "t", "b" for left, right, top, bottom)
        sigma_max: maximum damping coefficient in sponge layer
        snapshot_interval: time interval for saving snapshots (if None, only save final state)
        source: optional callable S(t) returning a (Ny, Nx) array that is added to the eta equation,
                e.g. from create_exact_unidirectional_source or create_spectral_source
        callback: optional callable callback(t, eta, phi), called after every time step. Handy for
                  recording time series in a subdomain without storing full snapshots.
    
    Returns:
        t_snapshots: list of time points for saved snapshots
        eta_snapshots: list of surface elevation arrays at saved snapshots
        phi_snapshots: list of surface potential arrays at saved snapshots
        energy_density_snapshots: list of energy density arrays at saved snapshots
        
    """
    
    X, Y, dx, dy = build_grid(Lx, Ly, Nx, Ny)
    Kx, Ky, K = build_wavenumbers(Lx, Ly, Nx, Ny)
    g = 9.81
    # Initial conditions
    eta = eta0
    phi = phi0
    Gphi0 = apply_dtn(phi, topo_data)
    Ux, Uy = U
    # add div(U)*eta term as a source to rhs in RK4
    divU = np.gradient(Ux,axis = 1)/dx + np.gradient(Uy,axis = 0)/dy
    sigma = build_sponge_sides(Lx, Ly, X, Y, sponge_frac=sponge_frac, sigma_max=sigma_max, sides = sponge_sides)
    
    n_steps = int(np.round(T_final / dt))
    
    if snapshot_interval is None:
        snapshot_every = n_steps
    else:
        snapshot_every = max(int(np.round(snapshot_interval / dt)), 1)

   
    #initialize snapshots
    
    t = 0.0
    eta_snapshots = [eta0.copy()]
    phi_snapshots = [phi0.copy()]
    t_snapshots = [t]
    energy_density_snapshots = [0.5*g*eta0.copy()**2 + 0.5*phi0.copy()*Gphi0.copy()] 
   
    for n in range(1, n_steps + 1):
        # Step from t to t + dt
        eta, phi = step_system_rk4(eta, phi, dt, Ux, Uy, divU, g, Kx, Ky, sigma, topo_data, source=source, t=t)
        t = n*dt
        
        if callback is not None:
            callback(t, eta, phi)
        
        # Save snapshots
        if n % snapshot_every == 0 or n == n_steps:
    
            Gphi = apply_dtn(phi, topo_data)
            energy_density = 0.5 * (g * eta**2 + phi * Gphi)    
            eta_snapshots.append(eta.copy())
            phi_snapshots.append(phi.copy())
            energy_density_snapshots.append(energy_density.copy())
            
            t_snapshots.append(t)
            print(f"Time: {t:.2f}/{T_final:.2f}", end='\r')

    return np.array(t_snapshots), eta_snapshots, phi_snapshots, energy_density_snapshots

# ============================================================
# 4. Utilities and Plots
# ============================================================



def wave_plots(eta_list, n_plots, X, Y, Lx, Ly, t_list, title=r"Surface Elevation $\eta(x,y)$", cmap="default"):
    
    total_snaps = len(t_list)
    n_plots = min(n_plots, total_snaps)
    
    indices = np.linspace(0, total_snaps - 1, n_plots, dtype=int)
    
    ncols = min(n_plots, 3)
    nrows = math.ceil(n_plots / ncols)
    
    domain_aspect = Ly / Lx  
    col_width = 4.5
    row_height = (col_width * domain_aspect) + 0.8 
    
    if cmap == "default":
        colors = ["#08306b", "#ffffff", "#41b6c4"] 
        Cmap = water_cmap = LinearSegmentedColormap.from_list("WaterWave", colors)
    
    else: Cmap = cmap
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(col_width * ncols, row_height * nrows), 
                             squeeze=False, layout="constrained")
    
    fig.suptitle(title, fontsize=16)
    
    vmax = np.max(np.abs(eta_list[0]))
    vmin = -vmax
    
    flat_axes = axes.flatten()
    
    for i, ax in enumerate(flat_axes):
        if i < n_plots:
            idx = indices[i]
            c = ax.pcolormesh(X, Y, eta_list[idx], shading="auto", cmap=Cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"$t = {t_list[idx]:.2f}$", fontsize=11)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect('equal')
        else:
            ax.axis('off')

    cbar = fig.colorbar(c, ax=axes, location='right', shrink=0.85)
    cbar.set_label(r"$\eta(x,y)$", fontsize=12)
    
    plt.show()
# ============================================================
# 5. Continuity/Transport Equations 
# ============================================================


def rhs_advection(t, u, Vx, Vy, g, KX, KY):
    """
    Computes the right-hand side of the PDE:
    du/dt = - (V dot grad(u)) + g(x)*u
    """
   
    du_dx, du_dy = spectral_gradient(u, KX, KY)
    
   
    advection = -(Vx * du_dx + Vy * du_dy)
    
    
    source = g * u
    
    return advection + source


def solve_advection_rk4(E0, V, g, Lx, Ly, T_final, dt, snapshot_interval=1.0):
    """
    Solves du/dt + V.grad(u) = gu using RK4 time stepping and Spectral gradients.
    
    Args:
        e0 : Initial condition (2D array)
        V : Tuple (Vx, Vy) of velocity fields (2D arrays)
        g : Source function field (2D array)
        Lx, Ly : Domain dimensions
        T_final : Total simulation time
        dt : Time step
        
    Returns:
        history : List of u fields at stored time steps
        times : List of time points
    """
    Ny, Nx = E0.shape
    X, Y,_,_ = build_grid(Lx, Ly, Nx, Ny)
    KX, KY,_ = build_wavenumbers(Lx, Ly, Nx, Ny)
    print("nx, ny =", Nx, Ny)
    Vx, Vy = V
    
    E = E0.copy()
    t = 0.0
    n_steps = int(np.round(T_final / dt))
    
    E_snapshots = [E.copy()]
    times = [t]
    
    # Time Stepping Loop
    for n in range(n_steps + 1):
        # RK4 Step
        k1 = rhs_advection(t, E, Vx, Vy, g, KX, KY)
        k2 = rhs_advection(t + 0.5*dt, E + 0.5*dt*k1, Vx, Vy, g, KX, KY)
        k3 = rhs_advection(t + 0.5*dt, E + 0.5*dt*k2, Vx, Vy, g, KX, KY)
        k4 = rhs_advection(t + dt, E + dt*k3, Vx, Vy, g, KX, KY)
        
        E = E + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Save snapshots
        if n % int(snapshot_interval / dt) == 0 or n == n_steps:
            E_snapshots.append(E.copy())
            times.append(t)
            print(f"Time: {t:.2f}/{T_final:.2f}", end='\r')
        
        t += dt
          
    return E_snapshots, times

### Schrödinger-type PDE with variable coefficients and spectral derivatives

def spectral_gradient_imag(f, KX, KY):
    """
    Computes gradients using FFT.
    KX, KY are meshgrids of wavenumbers.
    """
    f_hat = fft2(f)
    
    # Derivative in Fourier space is multiplication by i*k
    dfdx_hat = 1j * KX * f_hat
    dfdy_hat = 1j * KY * f_hat
    
    dfdx = ifft2(dfdx_hat)
    dfdy = ifft2(dfdy_hat)
    
    return dfdx, dfdy

def spectral_divergence(fx, fy, KX, KY):
    """
    Computes gradients using FFT.
    KX, KY are meshgrids of wavenumbers.
    """
    fx_hat = fft2(fx)
    fy_hat = fft2(fy)
    
    # Derivative in Fourier space is multiplication by i*k
    dfdx_hat = 1j * KX * fx_hat
    dfdy_hat = 1j * KY * fy_hat
    
    dfdx = ifft2(dfdx_hat)
    dfdy = ifft2(dfdy_hat)
    
    return dfdx + dfdy

def rhs_schrodinger(t, u, Vx, Vy, g, KX, KY, Dxx, Dyy, Dxy):
    """
    Computes the right-hand side of the PDE:
    du/dt = - (V dot grad(u)) + g(x)*u
    """
    # 1. Compute gradients via FFT (Non-dissipative spatial discretization)
    du_dx, du_dy = spectral_gradient_imag(u, KX, KY)
    
    # 2. Compute Advection term: - (Vx * du/dx + Vy * du/dy)
    advection = -(Vx * du_dx + Vy * du_dy)
    
    # 3. Compute Schrödinger term: Dxx * d^2u/dx^2 + Dyy * d^2u/dy^2 + 2*Dxy * d^2u/dxdy
    Fx = Dxx * du_dx + Dxy * du_dy
    Fy = Dxy * du_dx + Dyy * du_dy
    
    dispersion = 0.5*1j*spectral_divergence(Fx,Fy,KX,KY)
    
    # 3. Compute Source/Growth term
    source = g * u
    
    return advection + source - dispersion


def solve_schrodinger_rk4(A0, V, Dxx, Dyy, Dxy, g, Lx, Ly, T_final, dt, snapshot_interval=1.0):
    """
    Solves du/dt + V.grad(u) = gu using RK4 time stepping and Spectral gradients.
    
    Parameters:
        e0 : Initial condition (2D array)
        V : Tuple (Vx, Vy) of velocity fields (2D arrays)
        Dxx, Dyy, Dxy : Diffusion coefficients for Schrödinger term
        g : Source function field (2D array)
        Lx, Ly : Domain dimensions
        T_final : Total simulation time
        dt : Time step
        
    Returns:
        history : List of u fields at stored time steps
        times : List of time points
    """
    Ny, Nx = A0.shape
    X, Y,_,_ = build_grid(Lx, Ly, Nx, Ny)
    KX, KY,_ = build_wavenumbers(Lx, Ly, Nx, Ny)
    Vx, Vy = V
    
    A = A0.copy()
    t = 0.0
    n_steps = int(np.round(T_final / dt))
    
    A_snapshots = [A.copy()]
    times = [t]
    
    # Time Stepping Loop
    for n in range(n_steps + 1):
         
        k1 = rhs_schrodinger(t, A, Vx, Vy, g, KX, KY,  Dxx, Dyy, Dxy)
        k2 = rhs_schrodinger(t + 0.5*dt, A + 0.5*dt*k1, Vx, Vy, g, KX, KY, Dxx, Dyy, Dxy)
        k3 = rhs_schrodinger(t + 0.5*dt, A + 0.5*dt*k2, Vx, Vy, g, KX, KY, Dxx, Dyy, Dxy)
        k4 = rhs_schrodinger(t + dt, A + dt*k3, Vx, Vy, g, KX, KY, Dxx, Dyy, Dxy)
        
        A = A + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Save snapshots
        if n % int(snapshot_interval / dt) == 0 or n == n_steps:
            A_snapshots.append(A.copy())
            times.append(t)
            print(f"Time: {t:.2f}/{T_final:.2f}", end='\r')
        
        t += dt
          
    return A_snapshots, times

def plot_current(Ux, Uy, X, Y, title="Current Field"):
    
    plt.figure(figsize=(10, 5))
    speed = np.sqrt(Ux**2 + Uy**2)
    plt.pcolormesh(X, Y, speed, cmap='viridis', shading='auto')
    plt.colorbar(label='Vector Magnitude')
    plt.streamplot(X, Y, Ux, Uy, color='white', density=1.0, linewidth=0.6)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal')
    plt.show()
    


    
def get_k(x,y,U,V,D,nb_rays,wave_period,initial_location,wave_direction,T=80,nt=1200):
    
    import xarray as xa
    import sys,os

    testdir = os.path.dirname(os.getcwd() + '/')
    srcdir = '..'

    sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))
    from src.ocean_wave_tracing import Wave_tracing



    X0, XN = x[0] , x[-1]
    Y0, YN = y[0] , y[-1] 
    nx = len(x)
    ny = len(y)
    dx=dy=x[1]-x[0]

    i_w_side = 'left' # Initial wave propagation side

   
    ###
    # Ray tracing
    ###
    print('Performing ray tracing')
    wt = Wave_tracing(U, V, 
                      nx, ny, nt,
                      T, dx, dy, 
                      nb_wave_rays=nb_rays,
                      domain_X0=X0, domain_XN=XN,
                      domain_Y0=Y0, domain_YN=YN, 
                      d=D
                     )
    ipx = initial_location[0]
    ipy = initial_location[1]
    
    print("X0,Y0", ipx[0],ipy[nb_rays//2])
    if ipx is None:
        i_w_side = 'left' # Initial wave propagation side
        wt.set_initial_condition(wave_period=wave_period, theta0=wave_direction,
                         incoming_wave_side=i_w_side)
    else:
        assert len(ipx)==nb_rays

    #    # make sure initial directions are array of size nb_rays
    if not type(wave_direction)==np.ndarray:
            wave_direction = np.ones(nb_rays)*wave_direction
        
    
    wt.set_initial_condition(wave_period=wave_period, theta0=wave_direction,
                                 ipx=ipx,ipy=ipy)

    wt.solve()

    ###
    # Grid wave number values
    ###
    from scipy.interpolate import griddata, RBFInterpolator

    grid_x,grid_y=np.meshgrid(wt.x,wt.y)
    step=5
    print(step)
    points = np.array([wt.ray_x.ravel()[0::step],wt.ray_y.ravel()[0::step]])
    values_k = wt.ray_k.ravel()[0::step]
    values_kx = wt.ray_kx.ravel()[0::step]
    values_ky = wt.ray_ky.ravel()[0::step]
    # downsample for RBF
   
    grid_k = griddata(points.T,values_k, 
                         (grid_x, grid_y), method='linear',fill_value=wt.ray_k[0,0])

    grid_kx = griddata(points.T,values_kx, 
                         (grid_x, grid_y), method='linear',fill_value=wt.ray_kx[0,0])

    grid_ky = griddata(points.T,values_ky,
                              (grid_x,grid_y), method='linear',fill_value=wt.ray_ky[0,0])
                       
    
    return (wt, grid_kx, grid_ky, grid_k)

def intrinsic_wave_quantities(kx,ky,b,X,Y):
    
    ###
    # Compute intrinsic frequency and group velocity for given wavenumber and current fields.
    ###
    dx = X[0,1]-X[0,0]
    dy = Y[1,0]-Y[0,0]

    g = 9.81
    k = np.sqrt(kx**2 + ky**2)
    sigma = np.sqrt(g * k * np.tanh(k * b))
    Cp = sigma / k
    Cg_x = (0.5*Cp)*(1 + 2 * k * b / np.sinh(2 * k * b)) * (kx / k) 
    Cg_y = (0.5*Cp)*(1 + 2 * k * b / np.sinh(2 * k * b)) * (ky / k) 
    grad_x_sigma = np.gradient(sigma, axis=1) / dx
    grad_y_sigma = np.gradient(sigma, axis=0) / dy
    divCg = np.gradient(Cg_x, axis=1) / dx + np.gradient(Cg_y, axis=0) / dy
    
    # Diffraction matrix
    Cg =  (0.5*Cp) * (1 + 2 * k * b / np.sinh(2 * k * b)) 
    dCg = (g*b*(1/np.cosh(b*k)**2)*(1 - b*k*np.tanh(b*k)) -Cg**2)/sigma 
    C1 = Cg/k
    C2 = dCg - C1
    
    Dxx = C1 + C2 * (kx**2 / k**2)
    Dyy = C1 + C2 * (ky**2 / k**2)
    Dxy = C2*(kx*ky / k**2)
       
    return sigma, Cg_x, Cg_y, divCg, grad_x_sigma, grad_y_sigma, Dxx, Dyy, Dxy



# ============================================================
# 6. Wave sources (internal wavemaker)
# ============================================================
#
# The source S(t, X) is added to the eta equation. For a monochromatic component we use
# S = Re( s(X) exp(-i omega t) ), where s is built along a beam in direction theta:
#
#     s(x') = Lambda * exp(i k (x' - x_c)) * prod_j (d/dx' - i(k_j - k)) / (i(k - k_j)) B(x') * W(y'),
#
# with B a C-infinity bump, k_j all the other roots of the dispersion relation (so the source
# radiates nothing at those wavenumbers), W a transverse window and Lambda = 2 a |C_g(k)| / int B
# so that the radiated wave has amplitude a. See notebooks/example2_wavemaker.ipynb.


def dispersion_roots(omega, U, H, g=9.81, k_max=None):
    """
    All real roots k of the 1D dispersion relation (omega - U k)^2 = g|k| tanh(H|k|),
    for constant current U (along the beam) and depth H.
    """
    if k_max is None:
        k_max = 10*omega**2/g + 1.0
        if U != 0:
            k_max = max(k_max, 10*(omega + np.sqrt(g*H))**2/(g*U**2), 10*g/U**2)

    def D(k):
        return (omega - U*k)**2 - g*np.abs(k)*np.tanh(H*np.abs(k))

    # log-spaced scan on each side of zero, so both small and very large roots are caught
    q = np.logspace(-8, np.log10(k_max), 20000)
    roots = []
    for k_grid in (q, -q):
        D_vals = D(k_grid)
        for i in np.where(np.diff(np.sign(D_vals)) != 0)[0]:
            roots.append(sp.optimize.brentq(D, k_grid[i], k_grid[i+1], xtol=1e-14, rtol=1e-14))
    return np.sort(np.array(roots))


def group_velocity(k, omega, U, H, g=9.81):
    """ Total group velocity d omega/dk = U + d sigma/dk on the branch through (k, omega). """
    abs_k = np.abs(k)
    tanh_kH = np.tanh(abs_k*H)
    dsigma2_dk = g*np.sign(k)*(tanh_kH + abs_k*H*(1 - tanh_kH**2))
    return U + dsigma2_dk/(2*(omega - U*k))


def _current_along(U, theta):
    """ Projection of a constant current U (scalar = x-component, or (Ux, Uy)) on direction theta. """
    Ux, Uy = (U, 0.0) if np.isscalar(U) else U
    return Ux*np.cos(theta) + Uy*np.sin(theta)


def wave_frequency(k, theta, U, H, g=9.81):
    """ Absolute frequency omega = sigma(k) + U.k of a wave with wavenumber k > 0 in direction theta. """
    return np.sqrt(g*k*np.tanh(k*H)) + _current_along(U, theta)*k


def _bump_derivatives(u, n):
    """
    Derivatives d^m/du^m of the bump exp(-1/(1-u^2)) for m = 0..n, for |u| < 1.
    Each derivative is exp(-y) times a polynomial in u and y = 1/(1-u^2), and we
    track the coefficients of that polynomial using dy/du = 2 u y^2.
    """
    y = 1.0/(1.0 - u**2)
    base = np.exp(-y)
    terms = {(0, 0): 1.0}   # (power of u, power of y) -> coefficient
    derivs = []
    for m in range(n + 1):
        poly = sum(c*u**pu*y**py for (pu, py), c in terms.items())
        derivs.append(base*poly)
        new_terms = {}
        for (pu, py), c in terms.items():
            if pu > 0:
                new_terms[(pu - 1, py)] = new_terms.get((pu - 1, py), 0.0) + c*pu
            if py > 0:
                new_terms[(pu + 1, py + 1)] = new_terms.get((pu + 1, py + 1), 0.0) + 2.0*c*py
            new_terms[(pu + 1, py + 2)] = new_terms.get((pu + 1, py + 2), 0.0) - 2.0*c
        terms = new_terms
    return derivs


_BUMP_INTEGRAL = sp.integrate.quad(lambda u: np.exp(-1.0/(1.0 - u**2)), -1, 1)[0]   # ~0.444


def _transverse_window(Y_rot, length, taper):
    """ Window along the source line: 1 in the middle, cosine taper of width taper at the ends. """
    if length is None:
        return np.ones_like(Y_rot)
    d = 0.5*length - np.abs(Y_rot)        # distance to the end of the line
    if taper <= 0:
        return (d >= 0).astype(float)
    return 0.5*(1 - np.cos(np.pi*np.clip(d/taper, 0, 1)))


def unidirectional_source_field(X, Y, k, U, H, center, theta=0.0, a=1.0, phase=0.0,
                                width=None, length=None, taper=None, g=9.81):
    """
    Complex spatial amplitude s(X) of a source that radiates a single plane wave
    a*cos(k (x' - x_c) - omega t + phase) in direction theta, and nothing else.
    
    Args:
        X, Y: grid
        k: target wavenumber (> 0)
        U: constant current, scalar (x-component) or tuple (Ux, Uy)
        H: constant depth
        center: (x_c, y_c), centre of the source
        theta: propagation direction (radians, from the x-axis)
        a, phase: amplitude and phase of the radiated wave
        width: length of the source support along the beam (default: one wavelength)
        length: length of the source line across the beam (default: infinite)
        taper: width of the cosine taper at each end of the line (default: length/5)
        
    Returns: (s, omega), with S(t, X) = Re(s exp(-i omega t)).
    """
    k = abs(k)
    U_eff = _current_along(U, theta)
    sigma_k = np.sqrt(g*k*np.tanh(k*H))
    omega = sigma_k + U_eff*k
    if omega <= 0:
        raise ValueError(f"Wave with k={k:.3f} in direction {theta:.2f} is blocked by the current (U_eff={U_eff:.3f}).")
    cg = group_velocity(k, omega, U_eff, H, g)
    if cg <= 0:
        raise ValueError(f"Wave with k={k:.3f} in direction {theta:.2f} has C_g={cg:.3f} <= 0, so its energy goes the wrong way.")

    # The modes we want to kill: every other real root of the dispersion relation along the beam
    roots = dispersion_roots(omega, U_eff, H, g)
    unwanted = roots[np.abs(roots - k) > 1e-8*max(1.0, k)]

    # Operator prod_j (d/dx - i(k_j - k)) / (i(k - k_j)) as polynomial coefficients in d/dx
    coeffs = np.polynomial.polynomial.polyfromroots(1j*(unwanted - k))
    if len(unwanted) > 0:
        coeffs = coeffs/np.prod(1j*(k - unwanted))

    # Rotated coordinates: x' along the beam, y' across it, both relative to the centre
    x_c, y_c = center
    X_rot = (X - x_c)*np.cos(theta) + (Y - y_c)*np.sin(theta)
    Y_rot = -(X - x_c)*np.sin(theta) + (Y - y_c)*np.cos(theta)

    if width is None:
        width = 2*np.pi/k
    R = 0.5*width
    inside = np.abs(X_rot) < R
    u = X_rot[inside]/R

    # Apply the operator to the bump B(x'/R) (chain rule gives the 1/R^m)
    dB = _bump_derivatives(u, len(coeffs) - 1)
    Bop = sum(c*dB[m]/R**m for m, c in enumerate(coeffs))

    # Amplitude so that the radiated wave has amplitude a: eta = s_hat(k) / (2 C_g)
    Lam = 2*a*np.abs(cg)/(_BUMP_INTEGRAL*R)

    s = np.zeros(X.shape, dtype=complex)
    s[inside] = Lam*np.exp(1j*(k*X_rot[inside] + phase))*Bop
    if taper is None and length is not None:
        taper = length/5
    s *= _transverse_window(Y_rot, length, taper)
    return s, omega


def _ramp(t, T_ramp):
    """ Smooth start-up from 0 to 1 over T_ramp. """
    if t <= 0:
        return 0.0
    if t >= T_ramp:
        return 1.0
    return 0.5*(1 - np.cos(np.pi*t/T_ramp))


def create_exact_unidirectional_source(X, Y, k, U, H, center, theta=0.0, a=1.0, phase=0.0,
                                       width=None, length=None, taper=None,
                                       ramp_periods=3.0, g=9.81):
    """
    Monochromatic wavemaker. Returns a callable F(t) to pass as source= to simulate_wave_system.
    It radiates a*cos(k (x' - x_c) - omega t + phase) along direction theta (x' = distance along the beam),
    with no wave going upstream. See unidirectional_source_field for the arguments.
    The source is ramped up smoothly over ramp_periods wave periods.
    F.omega holds the absolute frequency.
    """
    s, omega = unidirectional_source_field(X, Y, k, U, H, center, theta=theta, a=a, phase=phase,
                                           width=width, length=length, taper=taper, g=g)
    s_re, s_im = np.real(s), np.imag(s)
    T_ramp = ramp_periods*2*np.pi/omega

    def F(t):
        # Re(s exp(-i omega t))
        return _ramp(t, T_ramp)*(s_re*np.cos(omega*t) + s_im*np.sin(omega*t))

    F.omega = omega
    return F


def create_spectral_source(X, Y, k, theta, a, U, H, center, phase=None,
                           width=None, length=None, taper=None,
                           ramp_periods=3.0, seed=None, g=9.81):
    """
    Spectral wavemaker: superposition of unidirectional sources, one for each component.
    
    Args:
        k, theta, a: arrays (same length) of wavenumbers, directions (radians) and amplitudes
        phase: array of phases (default: random, uniform on [0, 2pi), drawn with seed)
        width: support along each beam, scalar or array (default: one wavelength of each component)
        other arguments as in unidirectional_source_field. All components share the same centre.
        
    Returns a callable F(t) for simulate_wave_system. The field radiated downstream is
        eta = sum_j a_j cos(k_j e_j.(X - X_c) - omega_j t + phase_j),  e_j = (cos theta_j, sin theta_j).
    F.omega and F.phase hold omega_j and phase_j.
    """
    k, theta, a = np.broadcast_arrays(np.atleast_1d(k), np.atleast_1d(theta), np.atleast_1d(a))
    n = len(k)
    if phase is None:
        phase = np.random.default_rng(seed).uniform(0, 2*np.pi, n)
    phase = np.broadcast_to(phase, (n,))
    widths = np.broadcast_to(np.array(width, dtype=object), (n,))

    fields, omegas = [], []
    for j in range(n):
        s, om = unidirectional_source_field(X, Y, k[j], U, H, center, theta=theta[j], a=a[j], phase=phase[j],
                                            width=widths[j], length=length, taper=taper, g=g)
        fields.append(s)
        omegas.append(om)
    fields = np.array(fields)
    omegas = np.array(omegas)
    T_ramp = ramp_periods*2*np.pi/omegas.min()

    def F(t):
        return _ramp(t, T_ramp)*np.real(np.tensordot(np.exp(-1j*omegas*t), fields, axes=1))

    F.omega = omegas
    F.phase = np.array(phase)
    return F
