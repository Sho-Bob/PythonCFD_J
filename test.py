import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

ti.init(arch=ti.gpu)  # Initialize Taichi with GPU support

# Define parameters
jmax = 101
dx = 0.01
dt = 0.02  # Adjust based on your CFL conditions
XS = 0.0
q1 = 1.0
q2 = 0.0

# Taichi fields
x = ti.field(dtype=ti.f32, shape=jmax)
q = ti.field(dtype=ti.f32, shape=jmax)
alf = ti.field(dtype=ti.f32, shape=jmax)

@ti.func
def compute_flux(i, flag_periodic):
    # Implement flux computation using Taichi
    qm = 0.5 * (q[i] + abs(q[i]))
    qp = 0.5 * (q[i + 1] - abs(q[i + 1]))
    alf[i] = max(0.5 * qm**2, 0.5 * qp**2)
    if flag_periodic and i == jmax - 1:
        qm = 0.5 * (q[i] + abs(q[i]))
        qp = 0.5 * (q[0] - abs(q[0]))
        alf[i] = max(0.5 * qm**2, 0.5 * qp**2)

@ti.kernel
def update_fields(flag_periodic: ti.i32):
    for i in range(jmax - 1):
        compute_flux(i, flag_periodic)
    # Handle the Neumann boundary condition at the last index
    alf[jmax - 1] = 0.5 * q[jmax - 1]**2

@ti.kernel
def initialize():
    for i in range(jmax):
        x[i] = XS + dx * i
        if x[i] < 0.5:
            q[i] = q1
        else:
            q[i] = q2

def main():
    initialize()
    flag_periodic = 1  # 1 for periodic, 0 for not

    for step in range(100):  # Number of time steps
        update_fields(flag_periodic)

    # Data transfer from Taichi to NumPy for plotting
    q_np = q.to_numpy()
    x_np = x.to_numpy()
    plt.figure(figsize=(10, 6))
    plt.plot(x_np, q_np, label='State at final step')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()