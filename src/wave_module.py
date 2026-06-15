import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftfreq
from numpy.linalg import cond, solve
from dataclasses import dataclass
import scipy as sp  
import xarray as xa
import sys,os




# ============================================================
# 1. Core numerics: grid, wavenumbers, sponge, upwind
# ============================================================

def build_grid(Lx, Ly, Nx, Ny):
    x = np.linspace(0.0, Lx, Nx, endpoint=False)
    y = np.linspace(0.0, Ly, Ny, endpoint=False)
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

    
    # 2. Grid & Wavenumber Setup (XY Indexing)
    dx, dy = Lx / Nx, Ly / Ny
    kx_1d = 2 * np.pi * fftfreq(Nx, d=dx)
    ky_1d = 2 * np.pi * fftfreq(Ny, d=dy)
    
    # indexing='xy' -> shape (Ny, Nx)
    KX, KY = np.meshgrid(kx_1d, ky_1d, indexing='xy')
    
    K_mod = np.sqrt(KX**2 + KY**2)
    K_mod[0, 0] = 1.0  # Avoid div by zero, masked later

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

def rhs_system(eta, phi, Ux, Uy, divU, g, Kx, Ky, sigma, topo_data):
    # Spectral gradients for high accuracy
    d_eta_dx, d_eta_dy = spectral_gradient(eta, Kx, Ky)
    d_phi_dx, d_phi_dy = spectral_gradient(phi, Kx, Ky)
    
    # Advection terms
    adv_eta = Ux * d_eta_dx + Uy * d_eta_dy
    adv_phi = Ux * d_phi_dx + Uy * d_phi_dy
    
    # Vertical velocity from DtN
    Gphi = apply_dtn(phi, topo_data)

    # Evolution equations with sponge layers
    eta_t = -adv_eta -divU*eta + Gphi - sigma * eta
    phi_t = -adv_phi - g * eta - sigma * phi

    return eta_t, phi_t

def step_system_rk4(eta, phi, dt, Ux, Uy,divU, g, Kx, Ky, sigma, topo_data):
    k1_eta, k1_phi = rhs_system(eta, phi, Ux, Uy,divU, g, Kx, Ky, sigma, topo_data)
    
    k2_eta, k2_phi = rhs_system(
        eta + 0.5*dt*k1_eta,
        phi + 0.5*dt*k1_phi,
        Ux, Uy, divU, g, Kx, Ky, sigma, topo_data
    )
    
    k3_eta, k3_phi = rhs_system(
        eta + 0.5*dt*k2_eta,
        phi + 0.5*dt*k2_phi,
        Ux, Uy, divU, g, Kx, Ky, sigma, topo_data
    )
    
    k4_eta, k4_phi = rhs_system(
        eta + dt*k3_eta,
        phi + dt*k3_phi,
        Ux, Uy, divU, g, Kx, Ky, sigma, topo_data
    )

    eta_new = eta + (dt/6.0) * (k1_eta + 2*k2_eta + 2*k3_eta + k4_eta)
    phi_new = phi + (dt/6.0) * (k1_phi + 2*k2_phi + 2*k3_phi + k4_phi)
    return eta_new, phi_new


def simulate_water_waves_constant(X, Y, eta0, phi0, b0, T, N_snaps, g=9.81):
    """
    Solves linear water waves with constant depth using an Integrating Factor method.
    This method provides exact linear phase evolution to avoid numerical dispersion.
    """
    Ny, Nx = X.shape
    Lx = X[0, -1] - X[0, 0] + (X[0, 1] - X[0, 0])
    Ly = Y[-1, 0] - Y[0, 0] + (Y[1, 0] - Y[0, 0])
    
    # 1. Wavenumbers and Dispersion Relation
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=Lx/Nx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=Ly/Ny)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    
    # omega^2 = g * k * tanh(k * b0)
    G0 = K * np.tanh(K * b0)
    omega = np.sqrt(g * G0)
    omega[0, 0] = 0.0 # Avoid division by zero for the mean mode
    
    # 2. Time Stepping Setup
    dt = T / 1000  # Default to 1000 steps for stability, adjust as needed
    t_snaps = np.linspace(0, T, N_snaps)
    results_eta = []
    
    # Pre-calculate the exact rotation terms for one time-step dt
    cos_wt = np.cos(omega * dt)
    sin_wt = np.sin(omega * dt)
    
    # Transfer coefficients (used for rotating the phase)
    # We use small epsilon to avoid NaN at K=0
    eps = 1e-15
    S = G0 / (omega + eps)
    S_inv = g / (omega + eps)

    # 3. Main Loop
    eta_h = fft2(eta0)
    phi_h = fft2(phi0)
    
    snap_idx = 0
    t = 0.0
    num_steps = int(T / dt)
    
    for n in range(num_steps + 1):
        # Save snapshot if time matches
        if snap_idx < N_snaps and t >= t_snaps[snap_idx] - 1e-10:
            results_eta.append(np.real(ifft2(eta_h)))
            snap_idx += 1
            
        # EXACT rotation of phase in Fourier Space (no dispersion error)
        eta_next_h =  cos_wt * eta_h + S * sin_wt * phi_h
        phi_next_h = -S_inv * sin_wt * eta_h + cos_wt * phi_h
        
        eta_h, phi_h = eta_next_h, phi_next_h
        t += dt
        
    return t_snaps, np.array(results_eta)


def simulate_wave_system(
    Lx, Ly, Nx, Ny,
    T_final, dt,
    eta0, phi0,
    U,
    topo_data,
    bulk_data,
    energy_domain,
    g=9.81,
    sponge_frac=0.15,
    sponge_sides="lrtb", 
    sigma_max=1.0,
    snapshot_interval=None,
    compute_energy_integral=False
):
    X, Y, dx, dy = build_grid(Lx, Ly, Nx, Ny)
    Kx, Ky, K = build_wavenumbers(Lx, Ly, Nx, Ny)
    
    # Initial conditions
    eta = eta0
    phi = phi0
    Ux, Uy = U
    divU = np.gradient(Ux,axis = 1)/dx + np.gradient(Uy,axis = 0)/dy
    sigma = build_sponge_sides(Lx, Ly, X, Y, sponge_frac=sponge_frac, sigma_max=sigma_max, sides = sponge_sides)
    
    n_steps = int(np.round(T_final / dt))
    
    if snapshot_interval is None:
        snapshot_interval = n_steps

   
    
    # Create energy mask
    energy_mask = (X >= energy_domain[0]) & (X <= energy_domain[1]) & \
                  (Y >= energy_domain[2]) & (Y <= energy_domain[3])
    energy_mask = energy_mask.astype(float)
        
    t = 0.0
    eta_snapshots = [eta0.copy()]
    phi_snapshots = [phi0.copy()]
    t_snapshots = [t]
    energy_density_snapshots = [(g*eta0**2).copy()]
    energy_snapshots = [0]
    surface_integral_history = [0]
    bulk_integral_history = [0]
    
    for n in range(n_steps + 1):
        # Step
        eta, phi = step_system_rk4(eta, phi, dt, Ux, Uy, divU, g, Kx, Ky, sigma, topo_data)
        
        # Save snapshots
        if n % int(snapshot_interval / dt) == 0 or n == n_steps:
            # Energy Calculation (Hamiltonian density)
            # H = 0.5 * (g * eta^2 + phi * G[phi])
            Gphi = apply_dtn(phi, topo_data)
            energy_density = 0.5 * (g * eta**2 + phi * Gphi)
            if compute_energy_integral == True: 
                print("Computing energy integrals at time t =", t)
                surf_int, bulk_int = compute_energy_integrals_prop8(
                eta, phi, Kx, Ky, dx, X, 
                U_func=bulk_data['U_func'], 
                W_func=bulk_data['W_func'], 
                b_depth=bulk_data['b_depth']
            )
                surface_integral_history.append(surf_int)
                bulk_integral_history.append(bulk_int)
            
            # Integrate over domain
            total_energy = np.sum( energy_mask*energy_density) * dx * dy
        
            eta_snapshots.append(eta.copy())
            phi_snapshots.append(phi.copy())
            energy_density_snapshots.append(energy_density.copy())
            energy_snapshots.append(total_energy)
            t_snapshots.append(t)
            print(f"Time: {t:.2f}/{T_final:.2f}", end='\r')
        
        t += dt

    return np.array(t_snapshots), eta_snapshots, phi_snapshots, energy_density_snapshots, energy_snapshots, surface_integral_history, bulk_integral_history, (X, Y)

# ============================================================
# 4. Utilities and Plots
# ============================================================



def wave_plots(eta_list, n_plots, X, Y, Lx, Ly, t_list,title = "Surface Elevation η"):
    total_snaps = len(t_list)
    step = max(1, total_snaps // n_plots)
    
    for idx in range(0, total_snaps, step):
        plt.figure(figsize=(10, 3))
        plt.pcolormesh(X, Y, eta_list[idx], shading="auto", cmap='RdBu_r',vmin=-np.max(np.abs(eta_list[0])), vmax=np.max(np.abs(eta_list[0])))
        plt.colorbar(label=r"$\eta(x,y)$")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"{title} t = {t_list[idx]:.2f}")
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.show()

# ============================================================
# 5. Continuity/Transport Equations 
# ============================================================


def get_spectral_grid(Lx, Ly, Nx, Ny):
    """
    Generates grid and wavenumbers for spectral methods.
    Uses 'ij' (matrix) indexing for consistency with linear algebra.
    """
    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Wavenumbers (frequencies)
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=Lx/Nx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=Ly/Ny)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    
    return X, Y, KX, KY


def rhs_advection(t, u, Vx, Vy, g, KX, KY):
    """
    Computes the right-hand side of the PDE:
    du/dt = - (V dot grad(u)) + g(x)*u
    """
    # 1. Compute gradients via FFT (Non-dissipative spatial discretization)
    du_dx, du_dy = spectral_gradient(u, KX, KY)
    
    # 2. Compute Advection term: - (Vx * du/dx + Vy * du/dy)
    advection = -(Vx * du_dx + Vy * du_dy)
    
    # 3. Compute Source/Growth term
    source = g * u
    
    return advection + source


def solve_advection_rk4(E0, V, g, Lx, Ly, T_final, dt, snapshot_interval=1.0):
    """
    Solves du/dt + V.grad(u) = gu using RK4 time stepping and Spectral gradients.
    
    Parameters:
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


def generate_field_with_constraints(f_func, U_target, X0_target, Lx, Ly, Nx, Ny):
    """
    Generates a vector field U(x,y) such that:
      1. div(U) = f_func(x,y)
      2. U(X0_target) = U_target
    
    Parameters:
        f_func (function): The divergence source function f(X, Y).
        U_target (tuple): Desired velocity vector (u, v) at the target point.
        X0_target (tuple): Target location (x0, y0).
        Lx, Ly, Nx, Ny: Domain parameters.
    """
    # 1. Setup Grid
    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    # 2. Compute Divergence Source f(X)
    f_val = f_func
    
    # Enforce solvability for periodic domain (Net flux must be 0)
    f_mean = np.mean(f_val)
    if abs(f_mean) > 1e-10:
        print(f"Adjusting source mean by {-f_mean:.2e} to allow periodic solution.")
        f_val = f_val - f_mean

    # 3. Solve Poisson: Laplacian(phi) = f
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=Lx/Nx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=Ly/Ny)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0  # Avoid div/0
    
    f_hat = np.fft.fft2(f_val)
    Phi_hat = -f_hat / K2
    Phi_hat[0, 0] = 0.0
    
    # 4. Compute Raw Gradient Field (V_raw = grad(phi))
    U_hat = 1j * KX * Phi_hat
    V_hat = 1j * KY * Phi_hat
    
    U_raw = np.real(np.fft.ifft2(U_hat))
    V_raw = np.real(np.fft.ifft2(V_hat))
    
    # 5. Determine Correction Constant C
    # Find indices closest to X0_target
    idx_x = int(round(X0_target[0] / Lx * Nx)) % Nx
    idx_y = int(round(X0_target[1] / Ly * Ny)) % Ny
    
    # Sample what we have
    u_current = U_raw[idx_x, idx_y]
    v_current = V_raw[idx_x, idx_y]
    
    # Calculate difference
    C_u = U_target[0] - u_current
    C_v = U_target[1] - v_current
    
    # 6. Apply Correction
    U_final = U_raw + C_u
    V_final = V_raw + C_v
    
    return  U_final, V_final, f_val

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

def compute_envelope_metrics(energy_density_snaps,
                             energy_density_snaps_wa,
                             energy_domain,
                             p,
                             X,
                             Y):
    
    diffEEwa = [a - b for a, b in zip(energy_density_snaps, energy_density_snaps_wa)]
    
    energy_mask = (X >= energy_domain[0]) & (X <= energy_domain[1]) & \
                    (Y >= energy_domain[2]) & (Y <= energy_domain[3])
    energy_mask = energy_mask.astype(float)
    
  
    e_wa_max = []
    e_wa_Lp = []
    
    for e in energy_density_snaps_wa:
        e_wa_max.append(np.max(np.abs(energy_mask*e)))
        e_wa_Lp.append((np.sum((energy_mask*e)**p))**(1/p))
        
    e_full_max = []
    e_full_Lp = []
    
    for e in energy_density_snaps:
        e_full_max.append(np.max(np.abs(energy_mask*e)))
        e_full_Lp.append((np.sum((energy_mask*e)**p)**(1/p)))
        
        
    e_wa_diff = []
    for e in diffEEwa:
        e_wa_diff.append(np.max(np.abs(energy_mask*e)))
        
   
    return e_wa_diff, e_wa_max, e_wa_Lp, e_full_max, e_full_Lp
