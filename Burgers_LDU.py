import numpy as np
import matplotlib.pyplot as plt

def init(q1, q2, XS, dx, jmax):
    x = np.linspace(XS, XS + dx * (jmax-1), jmax)
    # q = np.array([float(q1) if i < 0.0 else float(q2) for i in x])
    # for i in x:
    q = 1.0*np.sin(2*np.pi*x)
    return (x, q)

def Godnov(alf, q, dt,dx, jmax, flag_periodic):
    for j in range(0,jmax-1):
        qm = 0.5*(q[j]+np.abs(q[j]))
        qp = 0.5*(q[j+1]-np.abs(q[j+1]))
        alf[j] = max(0.5*qm**2,0.5*qp**2)
    alf[jmax-1] = 0.5*q[jmax-1]**2 # Neumann
    if flag_periodic:
        qm = 0.5*(q[jmax-1]+np.abs(q[jmax-1]))
        qp = 0.5*(q[0]+np.abs(q[0]))
        alf[jmax-1] = max(0.5*qm**2,0.5*qp**2)

def UPWIND1(alf, q, c, dt, dx, jmax):
    for j in range(0, jmax - 1):
        ur, ul = q[j+1], q[j]
        fr, fl = c * ur, c * ul
        # alf[j] is flux at j+1/2
        alf[j] = 0.5 * (fr + fl - abs(c) * (ur - ul))

def do_computing_LDU(x, q, c, dt, dx, jmax, nmax, ff, flag_p, interval=2, xlim=None):
    plt.figure(figsize=(7,7), dpi=100)
    plt.rcParams["font.size"] = 22
    plt.plot(x, q, marker='o', lw=2, label='t = 0')

    alf = np.zeros(jmax)
    dq = np.zeros(jmax)
    for n in range(1, nmax + 1):
        qold = q.copy()
        
        c_a = np.abs(q)
        c_p = 0.5 * (q + c_a)
        c_n = 0.5 * (q - c_a)
        nu_a = c_a * dt / dx
        nu_p = c_p * dt / dx
        nu_n = c_n * dt / dx
        
        ff(alf, qold, dt, dx, jmax, flag_p)
        # print(alf)
        R = np.append(0.0, np.diff(alf) / dx)
        # print(R)
        # periodic BC
        if flag_p:
            R[0] = (alf[jmax-1]-alf[0])/dx

        for j in range(1, jmax - 1):
            dq[j] = (-dt * R[j] + nu_p[j-1] * dq[j - 1]) / (1 + nu_a[j])
        dq[0] = (-dt*R[0]+nu_p[0]*dq[0])/(1+nu_a[0])
        if flag_p:
            dq[0] = (-dt*R[0])/(1+nu_a[0])

        for j in range(jmax - 2, -1, -1):
            dq[j] = dq[j] - nu_n[j+1] * dq[j + 1] / (1 + nu_a[j])
        if flag_p:
            dq[jmax-1] = dq[jmax-1] - nu_n[0]*dq[0]/(1+nu_a[jmax-1]) 

        for j in range(0, jmax - 1):
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
    dx = 0.01
    CFL = 0.2
    dt = CFL *dx/c
    flag_periodic = False

    jmax = 100
    nmax = 100

    XS = 0

    q1 = 1
    q2 = 0
    x, q = init(q1, q2, XS, dx, jmax)
    do_computing_LDU(x, q, c, dt, dx, jmax, nmax, Godnov, flag_periodic,interval=25, xlim=[-0.1, 1.1])

if __name__ == "__main__":
    main()