import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

jmax = 1001
dt = 0.0002

gamma = 1.4

PI = 1.0
RHOI = 1.0
UI = 0.0

PE = 0.1
RHOE = 0.1
UE = 0.0

xmin, xmid, xmax = 0.0, 0.5, 1.0
x = np.linspace(xmin, xmax, jmax)

dx = (xmax - xmin) / (jmax - 1)

dtdx = dt / dx

def init():
    Q = np.zeros([jmax, 3])

    Q[x <= xmid, 0] = RHOI
    Q[x <= xmid, 1] = RHOI * UI
    Q[x <= xmid, 2] = (PI / (gamma - 1.0) + 0.5 * RHOI * UI ** 2)

    Q[x > xmid, 0] = RHOE
    Q[x > xmid, 1] = RHOE * UE
    Q[x > xmid, 2] = (PE / (gamma - 1.0) + 0.5 * RHOE * UE ** 2)

    return Q

def calc_CFL(Q):
    rho, rhou, e = Q[:, 0], Q[:, 1], Q[:, 2]
    
    u = rhou / rho
    p = (gamma - 1.0) * (e - 0.5 * rho * u ** 2)
    
    c = np.sqrt(gamma * p / rho)
    sp = c + np.abs(u)
    return max(sp) * dtdx   

def Roe_flux(QL, QR, E):
    for j in range(jmax - 1):        
        rhoL, uL, pL = QL[j, 0], QL[j, 1], QL[j, 2]
        rhoR, uR, pR = QR[j + 1, 0], QR[j + 1, 1], QR[j + 1, 2]
        
        rhouL = rhoL * uL
        rhouR = rhoR * uR

        eL = pL / (gamma - 1.0) + 0.5 * rhoL * uL ** 2
        eR = pR / (gamma - 1.0) + 0.5 * rhoR * uR ** 2

        HL = (eL + pL) / rhoL
        HR = (eR + pR) / rhoR
        
        cL = np.sqrt((gamma - 1.0) * (HL - 0.5 * uL ** 2))
        cR = np.sqrt((gamma - 1.0) * (HR - 0.5 * uR ** 2))
                
        sqrhoL = np.sqrt(rhoL)
        sqrhoR = np.sqrt(rhoR)

        rhoAVE = sqrhoL * sqrhoR
        uAVE = (sqrhoL * uL + sqrhoR * uR) / (sqrhoL + sqrhoR)
        HAVE = (sqrhoL * HL + sqrhoR * HR) / (sqrhoL + sqrhoR) 
        cAVE = np.sqrt((gamma - 1.0) * (HAVE - 0.5 * uAVE ** 2))
        eAVE = rhoAVE * (HAVE - cAVE ** 2 / gamma)
        
        dQ = np.array([rhoR - rhoL, rhoR * uR - rhoL * uL, eR - eL])
        
        Lambda = np.diag([np.abs(uAVE - cAVE), 
                          np.abs(uAVE), 
                          np.abs(uAVE + cAVE)])
        
        b1 = 0.5 * (gamma - 1.0) * uAVE ** 2 / cAVE ** 2
        b2 = (gamma - 1.0) / cAVE ** 2

        R = np.array([[1.0, 1.0, 1.0],
                      [uAVE - cAVE, uAVE, uAVE + cAVE],
                      [HAVE - uAVE * cAVE, 0.5 * uAVE ** 2, HAVE + uAVE * cAVE]])
        
        Rinv = np.array([[0.5 * (b1 + uAVE / cAVE), -0.5 * (b2 * uAVE + cAVE), 0.5 * b2],
                         [1.0 - b1, b2 * uAVE, -b2],
                         [0.5 * (b1 - uAVE / cAVE), -0.5 * (b2 * uAVE - cAVE), 0.5 * b2]])
        
        AQ = R @ Lambda @ Rinv @ dQ
        
        EL = np.array([rhoL * uL, pL + rhouL * uL, (eL + pL) * uL])
        ER = np.array([rhoR * uR, pR + rhouR * uR, (eR + pR) * uR])
        
        E[j] = 0.5 * (ER + EL - AQ)

def minmod(x, y):
    sgn = np.sign(x)
    return sgn * np.maximum(np.minimum(np.abs(x), sgn * y), 0.0)

def MUSCL(Q, order, kappa):
    rho, rhou, e = Q[:, 0], Q[:, 1], Q[:, 2]
    
    Q[:, 1] = rhou / rho  # u
    Q[:, 2] = (gamma - 1.0) * (e - 0.5 * rho * Q[:, 1] ** 2) # p
    
    if order == 2 or order == 3:
        dQ = np.zeros([jmax, 3])
        for j in range(jmax - 1):
            dQ[j] = Q[j+1] - Q[j]
        
        b = (3.0 - kappa) / (1.0 - kappa)
        
        Dp = np.zeros([jmax, 3])
        Dm = np.zeros([jmax, 3])
        for j in range(1, jmax - 1):
            Dp[j] = minmod(dQ[j], b * dQ[j - 1])
            Dm[j] = minmod(dQ[j-1], b * dQ[j])
        Dp[0] = Dp[1]
        Dm[0] = Dm[1]
        
        QL = Q.copy()
        QR = Q.copy()
        for j in range(1, jmax - 1):
            QL[j] += 0.25 * ((1.0 - kappa) * Dp[j] + (1.0 + kappa) * Dm[j])
            QR[j] -= 0.25 * ((1.0 + kappa) * Dp[j] + (1.0 - kappa) * Dm[j])
        
    else:
        QL = Q.copy()
        QR = Q.copy()

    return QL, QR

def Roe_FDS(Q, order, kappa, nmax, print_interval):
    E = np.zeros([jmax, 3])
    results = []

    for n in range(nmax):
        if n % print_interval == 0:
            print(f'n = {n : 4d} : CFL = {calc_CFL(Q) : .4f}')
            results.append(Q.copy())

        Qold = Q.copy()
        
        coefs = [0.5, 1.0]
        for coef in coefs:
            QL, QR = MUSCL(Qold, order, kappa)
        
            Roe_flux(QL, QR, E)
            for j in range(1, jmax - 1):
                Qold[j] = Q[j] - coef * dtdx * (E[j] - E[j-1])
            
            Qold[0] = Q[0]
            Qold[-1] = Q[-1]
    
        Q[:] = Qold[:]

    return results

def update_plot(frame, x, line):
    line.set_ydata(frame[:, 0])
    return line,

if __name__ == "__main__":
    nmax = 1000
    print_interval = 25

    order = 3

    kappa = 0

    Q = init()
    results = Roe_FDS(Q, order, kappa, nmax, print_interval)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
    plt.rcParams["font.size"] = 22
    ax.set_xlabel('x')
    ax.set_ylabel(r'$\rho$')
    ax.grid(color='black', linestyle='dotted', linewidth=0.5)
    line, = ax.plot(x, results[0][:, 0], color='red', linewidth=1.5)

    ani = animation.FuncAnimation(
        fig, update_plot, frames=results, fargs=(x, line), blit=True, interval=200
    )

    plt.show()
                                  
                                  