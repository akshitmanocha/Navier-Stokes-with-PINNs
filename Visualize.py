import torch
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import warnings
from Model import PINN
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the data
data = scipy.io.loadmat('dataset/cylinder_wake.mat')

U_star = data['U_star']  # N x 2 x T
P_star = data['p_star']  # N x T
t_star = data['t']  # T x 1
X_star = data['X_star']  # N x 2

N = X_star.shape[0]
T = t_star.shape[0]

# Prepare the data
XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
TT = np.tile(t_star, (1, N)).T  # N x T

x = XX.flatten()[:, None]  # NT x 1
y = YY.flatten()[:, None]  # NT x 1
t = TT.flatten()[:, None]  # NT x 1

model = PINN()
model.load_state_dict(torch.load('weights/model.pt'))
model.eval()

def get_prediction(x, y, t):
    x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    y = torch.tensor(y, dtype=torch.float32, requires_grad=True)
    t = torch.tensor(t, dtype=torch.float32, requires_grad=True)
    
    res = model(x, y, t)
    psi, p = res[:, 0:1], res[:, 1:2]
    
    u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    v = -1. * torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
    
    return u.detach().numpy(), v.detach().numpy(), p.detach().numpy()

def plot_fields_side_by_side(u, v, p, title, timestep):
    field_titles = ['u (x-velocity)', 'v (y-velocity)', 'Pressure']
    fields = [u, v, p]
    
    plt.figure(figsize=(18, 5))
    
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        field_plot = np.reshape(fields[i], (N, T))[:, timestep]
        plt.scatter(X_star[:, 0], X_star[:, 1], c=field_plot, cmap='jet')
        plt.colorbar()
        plt.title(f'{field_titles[i]} at t = {t_star[timestep][0]:.2f}')
        plt.xlabel('x')
        plt.ylabel('y')
    
    plt.tight_layout()
    plt.show()
    
time_steps = [0, T//2, T-1] 

for step in time_steps:
    t_plot = np.ones_like(x) * t_star[step]
    u, v, p = get_prediction(x, y, t_plot)
    
    plot_fields_side_by_side(u, v, p, 'Field Comparison', step)
