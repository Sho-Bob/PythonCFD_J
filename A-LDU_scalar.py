import numpy as np
import matplotlib.pyplot as plt

def init(q1, q2, XS, dx, jmax):
    x = np.linspace(XS, XS + dx * (jmax-1), jmax)
    q = np.array([float(q1) if i < 0.0 else float(q2) for i in x])
    return (x, q)

def UPWIND1(alf, q, c, dt, dx, jmax):
    for j in range(0, jmax - 1):
        ur, ul = q[j+1], q[j]
        fr, fl = c * ur, c * ul
        alf[j] = 0.5 * (fr + fl - abs(c) * (ur - ul))

def do_computing_LDU(x, q, c, dt, dx, jmax, nmax, ff, interval=2, xlim=None):
    plt.figure(figsize=(7,7), dpi=100)
    plt.rcParams["font.size"] = 22
    plt.plot(x, q, marker='o', lw=2, label='t = 0')

    alf = np.zeros(jmax)
    dq = np.zeros(jmax)
    for n in range(1, nmax + 1):
        qold = q.copy()
        
        c_a = abs(c)
        c_p = 0.5 * (c + c_a)
        c_n = 0.5 * (c - c_a)
        nu_a = c_a * dt / dx
        nu_p = c_p * dt / dx
        nu_n = c_n * dt / dx
        
        ff(alf, qold, c, dt, dx, jmax)
        R = np.append(0.0, np.diff(alf) / dx)

        for j in range(1, jmax - 1):
            dq[j] = (-dt * R[j] + nu_p * dq[j - 1]) / (1 + nu_a)

        for j in range(jmax - 2, 0, -1):
            dq[j] = dq[j] - nu_n * dq[j + 1] / (1 + nu_a)

        for j in range(1, jmax - 1):
            q[j] = qold[j] + dq[j]
        
        if n % interval == 0:
            plt.plot(x, q, marker='o', lw=2, label=f't = {dt * n : .1f}')

    plt.grid(color='black', linestyle='dashed', linewidth=0.5)
    plt.xlabel('x')
    plt.ylabel('q')
    if xlim is not None:
        plt.xlim(xlim)
    plt.legend()
    plt.show()

def main():
    c = 1
    dt = 0.05
    dx = 0.1

    jmax = 70
    nmax = 50

    XS = -1

    q1 = 1
    q2 = 0
    x, q = init(q1, q2, XS, dx, jmax)
    do_computing_LDU(x, q, c, dt, dx, jmax, nmax, UPWIND1, interval=16, xlim=[-1, 5])

if __name__ == "__main__":
    main()