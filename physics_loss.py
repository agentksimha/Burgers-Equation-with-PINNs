import torch
def pde_loss(model,x,t,alpha=0.01):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x,t)
    alpha = 0.01
    u_t = torch.autograd.grad(u,t,grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(u_x) , create_graph=True)[0]
    loss_pde = torch.mean((u_t - alpha * u_xx)**2) #for burgers equation we can simply change this to u_t - u*u_x - V * u_xx where V is a constant
    return loss_pde

def total_loss(model, X_f, T_f, X_bc, T_bc, U_bc, X_ic, T_ic, U_ic, alpha=0.01):
    # PDE loss with gradients (requires create_graph=True)
    loss_pde = pde_loss(model, X_f, T_f, alpha)

    # BC/IC losses (no derivatives needed, so just MSE)
    u_bc = model(X_bc, T_bc)
    u_ic = model(X_ic, T_ic)
    loss_bc = torch.mean((u_bc - U_bc)**2)
    loss_ic = torch.mean((u_ic - U_ic)**2)

    total = loss_pde + loss_bc + loss_ic
    return total, loss_pde, loss_bc, loss_ic

