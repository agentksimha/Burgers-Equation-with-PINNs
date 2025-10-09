import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
from utils import true_solution
from scipy.special import iv

def plot_results(model, x, t, alpha=0.01, save_path=None):
    Nx, Nt = len(x), len(t)
    X, T = np.meshgrid(x, t)
    x_test = torch.tensor(X.flatten()[:, None], dtype=torch.float32)
    t_test = torch.tensor(T.flatten()[:, None], dtype=torch.float32)

    with torch.no_grad():
        U_pred = model(x_test, t_test).cpu().numpy().reshape(Nt, Nx)

    U_true = true_solution(X, T, alpha)
    #U_true = np.zeros_like(U_pred)
    #for i, ti in enumerate(t):
        #decay = np.exp(-nu * (np.pi * k) ** 2 * ti)
        #cos_terms = np.cos(np.pi * np.outer(k, x))
        #sin_terms = np.sin(np.pi * np.outer(k, x))
        #theta = I0 + 2 * np.sum(Ik[:, None] * cos_terms * decay[:, None], axis=0)
        #theta_x = -2 * np.sum(Ik[:, None] * (np.pi * k)[:, None] * sin_terms * decay[:, None], axis=0)
        #U_true[i, :] = -2 * nu * theta_x / theta
    #the above code is for calculating U_true for burgers equation 
    error = np.abs(U_pred - U_true)

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot_surface(X, T, U_true, cmap="viridis")
    ax1.set_title("True Solution")

    ax2 = fig.add_subplot(132, projection="3d")
    ax2.plot_surface(X, T, U_pred, cmap="plasma")
    ax2.set_title("PINN Predicted")

    ax3 = fig.add_subplot(133, projection="3d")
    ax3.plot_surface(X, T, error, cmap="inferno")
    ax3.set_title("Absolute Error")

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_zlabel("u(x,t)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_loss_curve(losses):
    plt.figure(figsize=(6,4))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.show()
