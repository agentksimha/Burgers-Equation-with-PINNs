Physics-Informed Neural Network for 1D Heat & Burgers’ Equation

This project implements a Physics-Informed Neural Network (PINN) to solve the 1D heat equation:

𝑢
𝑡
=
𝛼
𝑢
𝑥
𝑥
,
𝑥
∈
[
0
,
1
]
,
 
𝑡
∈
[
0
,
3600
]
u
t
	​

=αu
xx
	​

,x∈[0,1], t∈[0,3600]

and the 1D Burgers’ equation:

𝑢
𝑡
+
𝑢
 
𝑢
𝑥
=
𝜈
 
𝑢
𝑥
𝑥
,
𝑥
∈
[
0
,
1
]
,
 
𝑡
∈
[
0
,
𝑇
]
u
t
	​

+uu
x
	​

=νu
xx
	​

,x∈[0,1], t∈[0,T]
✳️ Features

Fully implemented in PyTorch

Solves PDEs without labeled data

Enforces boundary and initial conditions in loss

3D visualizations of true vs predicted solutions

Configurable NN architecture (neurons, layers, epochs)

Burgers’ Equation PINN

Problem:
I solved the 1D Burgers’ equation using a PINN. Burgers’ equation models nonlinear convection and diffusion phenomena.

Boundary & Initial Conditions:

Initial condition (IC): 
𝑢
(
𝑥
,
0
)
=
sin
⁡
(
𝜋
𝑥
)
u(x,0)=sin(πx)

Boundary conditions (BCs): 
𝑢
(
0
,
𝑡
)
=
𝑢
(
1
,
𝑡
)
=
0
u(0,t)=u(1,t)=0

Neural Network Architecture & Training:

Initial setup: Fully connected NN with 70 neurons per hidden layer, trained for 300 epochs

Results: Achieved reasonable approximation of the solution; loss curve(loss_curve) and solution surface(Solution_curve) are included in images.

Improved setup: Increased to 100 neurons per hidden layer, trained for 350 epochs

Results: Loss convergence and solution accuracy improved significantly(loss_curve1 and Solution_curve1); sharper gradients captured, residual errors reduced.

Visualizations:

Loss curves for both setups

Predicted vs true solution surface plots

Improvements visible in the closer match between PINN prediction and analytical solution