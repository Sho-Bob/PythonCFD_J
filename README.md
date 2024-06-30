# Added by Sho Wada

### Comment
Boundary condition may be wrong. Gotta fix it. I hope someone can learn how to advance numerical solutions either explicitly or implicitly in an Euler system.

## Sod Shock Tube Problem Solver

This repository contains a 1-dimensional numerical solver for the Sod shock tube problem using various numerical schemes, including the Roe Flux Difference Splitting (Roe FDS) method with Runge-Kutta (RK) and Lower-Upper Symmetric Gauss-Seidel (LU-SGS) schemes. You'll get the following result.
![Sod Shock Tube Problem](image/Sod_time_integration_RoeFDS.png)

## Overview

The solver initializes the computational domain and the initial conditions for density, velocity, and pressure. The domain is divided into a specified number of grid points, and initial conditions are set for the left and right regions of the shock tube.

## Features

- Initialization of the computational domain and initial conditions.
- Calculation of the Courant-Friedrichs-Lewy (CFL) condition for numerical stability.
- Roe Flux Difference Splitting (Roe FDS) method for flux computation.
- Monotonic Upstream-centered Scheme for Conservation Laws (MUSCL) for higher-order spatial accuracy.
- Jacobian matrix computation for the LU-SGS scheme.
- Support for both explicit and implicit time integration methods with first-order and second-order temporal accuracy.

## Getting Started

### Prerequisites

- Python 3.x
- NumPy
- Matplotlib

### Running the Solver

To run the solver and visualize the results, execute the following command:

```bash
python Euler_FDS_implicit_HO.py

