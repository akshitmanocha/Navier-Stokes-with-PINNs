import torch
from tqdm import tqdm
from Loss import total_loss

def train_LBFGS(model, x, y, t, u, v, optimizer, epochs):
    model.train()
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        def closure():
            optimizer.zero_grad()
            loss = total_loss(model, x, y, t, u, v)
            loss.backward()
            return loss
        
        loss = optimizer.step(closure)
        
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

    print("Training completed.")

    
def train_Adam(model, x, y, t, u, v, optimizer, epochs):
    model.train()
    for epoch in tqdm(range(epochs), desc="Training Progress"):
            optimizer.zero_grad()
            loss = total_loss(model, x, y, t, u, v)
            loss.backward()
            optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
    print("Training completed.")
