import torch
import torch.nn as nn

nu = 0.01

def predictions(model, x, y, t):
    res = model(x, y, t)
    psi, p = res[:, 0:1], res[:, 1:2]

    u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    v = -1.*torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    f = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    g = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    return u, v, p, f, g

def total_loss(model, x, y, t, u, v):
    mse_loss = nn.MSELoss()
    u_prediction, v_prediction, p_prediction, f_prediction, g_prediction = predictions(model, x, y, t)
    
    u_loss = mse_loss(u_prediction, u)
    v_loss = mse_loss(v_prediction, v)
    f_loss = mse_loss(f_prediction, torch.zeros_like(f_prediction))
    g_loss = mse_loss(g_prediction, torch.zeros_like(g_prediction))
    
    return u_loss + v_loss + f_loss + g_loss