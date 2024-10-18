import torch
import numpy as np
from Dataset import load_data, extract_data
from Model import PINN
from Train import train_LBFGS
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_train, y_train, t_train, u_train, v_train = load_data('dataset/cylinder_wake.mat')

    x, y, t, u, v = extract_data(x_train, y_train, t_train, u_train, v_train)
    x = torch.tensor(x, dtype=torch.float32, requires_grad=True).to(device)
    y = torch.tensor(y, dtype=torch.float32, requires_grad=True).to(device)
    t = torch.tensor(t, dtype=torch.float32, requires_grad=True).to(device)
    u = torch.tensor(u, dtype=torch.float32).to(device)
    v = torch.tensor(v, dtype=torch.float32).to(device)


    model = PINN().to(device)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=20, max_eval=20,
                              history_size=50, tolerance_grad=1e-07, tolerance_change=1e-09,
                              line_search_fn="strong_wolfe")
    
    
    train_LBFGS(model, x, y, t, u, v, optimizer, epochs=20000)
    torch.save(model.state_dict(), 'weights/model.pt')