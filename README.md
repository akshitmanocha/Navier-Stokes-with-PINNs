# Physics-Informed Neural Network (PINN) for Cylinder Wake Flow Simulation

## Introduction

This project implements a Physics-Informed Neural Network (PINN) to solve the flow around a cylinder, governed by the Navier-Stokes equations. PINNs are neural networks trained to satisfy both observed data and the underlying physical laws described by partial differential equations (PDEs). The Navier-Stokes equations describe fluid dynamics and are essential for modeling fluid flow in various fields, including engineering and physics. Traditional computational fluid dynamics (CFD) methods often require intensive computational resources, whereas PINNs provide a flexible approach that leverages neural networks to approximate solutions while implicitly satisfying these physical constraints.

## Mathematical Formulation

### Differential Equations

The Navier-Stokes equations for incompressible fluid flow in two dimensions are given by:

**Momentum equations:**

∂u/∂t + u∂u/∂x + v∂u/∂y = -∂p/∂x + ν(∂²u/∂x² + ∂²u/∂y²)

∂v/∂t + u∂v/∂x + v∂v/∂y = -∂p/∂y + ν(∂²v/∂x² + ∂²v/∂y²)

**Continuity equation (for incompressibility):**

∂u/∂x + ∂v/∂y = 0

Here, u and v are the velocity components, p is pressure, and ν is the kinematic viscosity.

### Boundary and Initial Conditions

The boundary conditions for the cylinder are specified at the surface and at the inlet and outlet of the domain. Typically, no-slip conditions are enforced at the cylinder surface (i.e., u=0, v=0), while flow velocity is specified at the domain's inlet and outlet.

### Traditional Approaches

Standard approaches to solve this system often include finite element or finite difference methods. However, such methods require high computational power, especially for complex domains. This project explores a PINN approach to directly approximate the solution without extensive discretization or computational overhead.

## PINN Design

### Neural Network Architecture

The PINN model is a fully connected neural network with the following architecture:

- **Input Layer:** Accepts three inputs (x,y,t)
- **Hidden Layers:** 9 layers with 20 neurons each and Tanh activation functions
- **Output Layer:** Outputs two values, stream function ψ and pressure p

This network structure is designed to balance expressiveness and computational efficiency.

### Training Process

We use two different optimizers in training:

1. **Adam Optimizer:** For initial training with a moderate learning rate, providing stability during the early stages
2. **LBFGS Optimizer:** A second optimizer for fine-tuning, enhancing convergence

The model is trained for 20,000 epochs with a learning rate of 0.1 for LBFGS, minimizing the loss between predicted and actual values for u, v, and p and enforcing compliance with the Navier-Stokes equations.

# Implementation

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

### Tools and Libraries Used

- Python with NumPy for data handling and mathematical operations
- PyTorch for building, training, and managing the neural network
- SciPy for loading MATLAB .mat files
- Matplotlib for visualizing flow field predictions

### Code Structure

- **Data Loading and Preparation:** `load_data()` loads and preprocesses flow data from the given .mat file
- **Model Definition:** `PINN` class defines the neural network architecture
- **Training:** `train_Adam()` and `train_LBFGS()` handle the optimization process
- **Prediction and Visualization:** `get_prediction()` and `plot_fields_side_by_side()` are used to predict flow fields and visualize results across time steps

### Challenges

A primary challenge involved tuning the model to satisfy both data points and the PDE constraints simultaneously. Balancing training stability while maintaining physical consistency required careful optimizer selection and parameter tuning.

## Results

The PINN model provides a close approximation to the true velocity and pressure fields in the cylinder wake. The model's predictions are visualized below for several time steps, with fields for:

- u - x-direction velocity
- v - y-direction velocity
- p - pressure

These plots demonstrate that the PINN can accurately capture the flow dynamics and produce results that align well with expected patterns.

## Conclusion

The PINN model effectively approximates the Navier-Stokes equations for the cylinder wake flow problem, achieving good accuracy in velocity and pressure predictions. The flexibility of PINNs makes them a valuable alternative to traditional CFD methods, particularly when computational resources are limited. However, challenges remain in optimizing model training for large-scale simulations.

## Future Improvements

Potential areas for improvement include:

- **Enhanced Network Architecture:** Experimenting with larger or deeper networks for more complex flows
- **Adaptive Loss Function Weighting:** Adjusting loss components adaptively to balance data and physics constraints
- **Hybrid Solvers:** Combining PINNs with traditional CFD solvers to leverage the strengths of both approaches

## References
The model architecture and training methodology are based on the research presented in the following paper and articles:
- [Research Paper](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- [Article](https://machinelearningmastery.com/bfgs-optimization-in-python/)




