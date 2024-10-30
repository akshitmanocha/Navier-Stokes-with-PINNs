# Physics-Informed Neural Networks for Navier-Stokes Equations

This project implements a Physics-Informed Neural Network (PINN) to solve the Navier-Stokes equations. PINNs are neural networks that are trained to satisfy governing physical equations as part of their loss function, combining traditional data-driven learning with physics-based constraints.

## Requirements

```
torch>=1.9.0
numpy>=1.19.2
matplotlib>=3.3.2
tqdm>=4.50.2
```

## Project Structure

```
├── loss.py           # Contains loss functions and training loops
├── model.py          # Neural network architecture
├── train.py          # Training functions 
└── dataset.py          # Dataloader file
```

## Key Features

- Implementation of physics-informed loss function
- Support for both LBFGS and Adam optimizers
- Automatic differentiation for computing derivatives
- Visualization utilities for results
- Custom neural network architecture optimized for fluid dynamics

## Code Overview

#### 1. Loss Function:
```python
def predictions(model, x, y, t):
    ...
    return u, v, p, f, g
```
The above prediction function predicts u v and p where are the values to be predicted and it also calculates f and g which are the physics loss which is to be minimized
f and g are calculated from the diffrential equation and torch.autograd is used to calculate the derivatives

```python
def total_loss(model, x, y, t, u, v):
    ...    
    return u_loss + v_loss + f_loss + g_loss
```
Here we have defined our loss function which is the sum of MSE loss and Physics Loss which is to be minimized

#### 2. Train function:
```python
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
```
In the above snippet we have defined the function for training of the model. Generally we use the LBFGS optmizer for training PINNs but we have trained the model using Adam optmizer, However you can use the LBFGS optimizer as well


## Technical Details

### Loss Function Components

The total loss function combines:
1. MSE loss between predicted and actual velocities (u, v)
2. Physics-informed loss terms (f, g) derived from Navier-Stokes equations

```python
total_loss = u_loss + v_loss + f_loss + g_loss
```

### Derivative Computation

Derivatives are computed using `torch.autograd.grad()` for:
- Spatial derivatives (∂/∂x, ∂/∂y)
- Temporal derivatives (∂/∂t)
- Second-order derivatives (∂²/∂x², ∂²/∂y²)

## Results

The model achieves accurate predictions of fluid flow fields while maintaining physical consistency through the enforcement of the Navier-Stokes equations.

## References
The model architecture and training methodology are based on the research presented in the following paper and articles:
- [Research Paper](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- [Article](https://machinelearningmastery.com/bfgs-optimization-in-python/)




