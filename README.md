****Physics-Informed Neural Network for 1D Heat & Burgers Equation


The 1D heat equation models flow of heat in systems like a 1D metal rod and Burgers Equation models nonlinear convection and diffusion phenomena.


This project implements a Physics-Informed Neural Network (PINN) to solve the heat-Equation 


u_t  = alpha * u_xx,  x E [0,1]  , t E [0,3600]

The analytical solution to heat equation with given boundary conditions is u = e^(-alpha * pi^2 *t) * sin(pi * x)

and Burgers_equation:


u_t + u * u_xx = v *u_xx

As the analytical solution of Burgers equation involves use of infinite sum  with Basel function we will use a approximate form of it with finite sum .

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





Neural Network Architecture & Training For heat equation:


Initial setup: Fully connected NN with 70 neurons per hidden layer, trained for 300 epochs with learning rate = 0.01 and alpha = 0.01





Improved setup: Increased to 100 neurons per hidden layer, trained for 350 epochs


Results: Loss convergence and solution accuracy improved significantly (loss_curve1 and Solution_curve1); sharper gradients captured, residual errors reduced.



Neural Network Architecture & Training  for Burgers Equation


For the Burgers’ equation, the PINN model was designed as a fully connected neural network that takes space (x) and time 


(t) as inputs and outputs the predicted solution u(x,t).

Initial setup (represented by loss_curves_burgers(1) and Solution_curves_burgers(1)):

120 neurons per hidden layer

350 training epochs

Learning rate: 0.01

Viscosity term (ν): 0.01

This configuration achieved smooth and accurate solution curves, capturing both the diffusion and nonlinear convection effects effectively.

Increased complexity(representedby loss_curves_burgers(3) and solution_curves_burgers(3)) :

130 neurons per hidden layer

420 training epochs (same learning rate and viscosity)

While this setup increased model capacity, it led to overfitting—the predicted solution curves began deviating from the analytical solution, especially at higher time steps.

The loss reduced numerically but the physical fidelity of the solution degraded.

Conclusion: The results highlight that increasing network size beyond a threshold can cause overfitting in PINNs, as the model starts fitting numerical noise instead of respecting the PDE’s underlying physics.

****Future Work:

To extend this study toward the inviscid Burgers’ equation (where viscosity ν = 0 and convection coefficient c = 1), the current PINN framework can be adapted into a Piecewise Shock Neural Network (PSNN). Unlike 

standard PINNs, which assume solution smoothness, PSNNs handle shock discontinuities by dividing the spatial domain into piecewise regions and enforcing the Rankine–Hugoniot jump condition at the shock 

interface. This allows accurate learning of discontinuous solutions and physically consistent shock propagation in the inviscid limit.





Improvements visible in the closer match between PINN prediction and analytical solution
