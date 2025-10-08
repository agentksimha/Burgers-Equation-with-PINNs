import torch
import numpy as np
import math

def true_solution(x, t, alpha=0.01):
    """Analytical solution for 1D Heat Equation u_t = alpha*u_xx with sin(πx) IC."""
    return np.exp(-alpha * math.pi**2 * t) * np.sin(math.pi * x)

import torch
import numpy as np

def generate_data(Nx=100, Nt=100, Nf=10000, x0=0, xmax=1, t0=0, tmax=3600):
    x = np.linspace(x0, xmax, Nx).reshape(-1,1).astype(np.float32)
    t = np.linspace(t0, tmax, Nt).reshape(-1,1).astype(np.float32)

    # Initial condition
    X_ic = torch.tensor(x, dtype=torch.float32)
    T_ic = torch.zeros_like(X_ic)
    U_ic = torch.sin(np.pi * X_ic)

    # Boundary condition
    T_bc = torch.tensor(t, dtype=torch.float32)
    X_bc_left = torch.zeros_like(T_bc)
    X_bc_right = torch.ones_like(T_bc)
    U_bc = torch.zeros_like(T_bc)
    X_bc = torch.cat([X_bc_left, X_bc_right], dim=0)
    T_bc = torch.cat([T_bc, T_bc], dim=0)
    U_bc = torch.cat([U_bc, U_bc], dim=0)

    # Collocation points for PDE residual (do NOT require grad here)
    x_f = np.random.uniform(x0, xmax, (Nf,1)).astype(np.float32)
    t_f = np.random.uniform(t0, tmax, (Nf,1)).astype(np.float32)
    X_f = torch.tensor(x_f, dtype=torch.float32)
    T_f = torch.tensor(t_f, dtype=torch.float32)

    # Normalize time (out-of-place)
    T_ic = T_ic / tmax
    T_bc = T_bc / tmax
    T_f = T_f / tmax

    return X_ic, T_ic, U_ic, X_bc, T_bc, U_bc, X_f, T_f, x, t
