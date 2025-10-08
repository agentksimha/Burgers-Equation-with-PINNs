Physics-Informed Neural Network for 1D Heat & Burgers’ Equation

The 1D heat equation models flow of heat in systems like a 1D metal rod and Burgers Equation models nonlinear convection and diffusion phenomena.

This project implements a Physics-Informed Neural Network (PINN) to solve the heat-Equation :

u_t  = alpha * u_xx,  x E [0,1]  , t E [0,3600]

and Burgers_equation:

u_t + u * u_xx = v *u_xx

Features

Fully implemented in PyTorch

Solves PDEs without labeled data

Enforces boundary and initial conditions in the loss

3D visualizations of true vs predicted solutions

Configurable neural network architecture (neurons, layers, epochs)


Boundary and intial conditions for both equations:

Intial conditions:

u(x,0) = sin(pi * x)

Boundary conditions:

u(0,t) = u(1,t) = 0





Neural Network Architecture & Training:

Initial setup: Fully connected NN with 70 neurons per hidden layer, trained for 300 epochs

Results: Achieved reasonable approximation of the solution; loss curve (loss_curve) and solution surface (Solution_curve) are included in images.

Improved setup: Increased to 100 neurons per hidden layer, trained for 350 epochs

Results: Loss convergence and solution accuracy improved significantly (loss_curve1 and Solution_curve1); sharper gradients captured, residual errors reduced.


Predicted vs true solution surface plots

Improvements visible in the closer match between PINN prediction and analytical solution
