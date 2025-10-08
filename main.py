import torch
from model import PINN
from utils import generate_data
from physics_loss import total_loss
from plot_utils import plot_results, plot_loss_curve

# ---- Data ----
X_ic, T_ic, U_ic, X_bc, T_bc, U_bc, X_f, T_f, x, t = generate_data()

# ---- Model ----
model = PINN(hidden_dim=100)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ---- Training ----
epochs = 350
print_every = 20
loss_history = []

for epoch in range(epochs):
    optimizer.zero_grad()
    total, lpde, lbc, lic = total_loss(model, X_f, T_f, X_bc, T_bc, U_bc, X_ic, T_ic, U_ic)
    total.backward()
    optimizer.step()
    loss_history.append(total.item())

    if (epoch + 1) % print_every == 0:
        print(f"Epoch {epoch+1}/{epochs} | Total: {total.item():.5e} | PDE: {lpde.item():.3e} | BC: {lbc.item():.3e} | IC: {lic.item():.3e}")

# ---- Plots ----
plot_loss_curve(loss_history)
plot_results(model, x, t)
