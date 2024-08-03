import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

jmax = 101
dt = 0.01

gamma = 1.4

PI = 1.0
RHOI = 1.0
UI = 0.0

PE = 0.1
RHOE = 0.1
UE = 0.0

xmin, xmid, xmax = 0.0, 0.5, 1.0 # Sod problem
# xmin, xmid, xmax = -4.0, 0.0, 4.0 # Shu Osher problem
x = np.linspace(xmin, xmax, jmax)

dx = (xmax - xmin) / (jmax - 1)

dtdx = dt / dx

def init_Sod():
    Q = np.zeros([jmax, 3])

    Q[x <= xmid, 0] = RHOI
    Q[x <= xmid, 1] = RHOI * UI
    Q[x <= xmid, 2] = (PI / (gamma - 1.0) + 0.5 * RHOI * UI ** 2)

    Q[x > xmid, 0] = RHOE
    Q[x > xmid, 1] = RHOE * UE
    Q[x > xmid, 2] = (PE / (gamma - 1.0) + 0.5 * RHOE * UE ** 2)

    return Q

def init_Shu_Osher():
    Q = np.zeros([jmax,3])
    rhoL = 3.857143
    uL = 2.629369
    pL = 10.333
    pR=1.0
    Q[x<-1.0,0] = rhoL 
    Q[x<-1.0,1] = rhoL*uL
    Q[x<-1.0,2] = (pL/(gamma-1.0)+ 0.5*rhoL*uL**2)
    # Q[x>=-4.0,0] = 1.0+0.2*np.sin(5.0*x) 
    Q[x >= -1, 0] = 1.0 + 0.2 * np.sin(5.0 * x[x >= -1])
    Q[x>=-1,1] = 0.0
    Q[x>=-1.0,2] = pR/(gamma-1.0)

    return Q

def calc_CFL(Q):
    rho, rhou, e = Q[:, 0], Q[:, 1], Q[:, 2]
    
    u = rhou / rho
    p = (gamma - 1.0) * (e - 0.5 * rho * u ** 2)
    
    c = np.sqrt(gamma * p / rho)
    sp = c + np.abs(u)
    return max(sp) * dtdx   

def HLLC(QL, QR, E):
    for j in range(jmax - 1):  
        rhoL, uL, pL = QL[j, 0], QL[j, 1], QL[j, 2]
        rhoR, uR, pR = QR[j + 1, 0], QR[j + 1, 1], QR[j + 1, 2]

        rhouL = rhoL * uL
        rhouR = rhoR * uR

        EL = pL / (gamma - 1.0) + 0.5 * rhoL * uL ** 2
        ER = pR / (gamma - 1.0) + 0.5 * rhoR * uR ** 2

        HL = (EL + pL) / rhoL
        HR = (ER + pR) / rhoR
        
        cL = np.sqrt((gamma - 1.0) * (HL - 0.5 * uL ** 2))
        cR = np.sqrt((gamma - 1.0) * (HR - 0.5 * uR ** 2))

        f_L = np.array([rhouL, rhouL * uL + pL, (EL + pL) * uL])
        f_R = np.array([rhouR, rhouR * uR + pR, (ER + pR) * uR])

        sL = min(uL-cL,uR-cR)
        sR = max(uL+cL,uR+cR)

        # Calculate star region values
        c_L = rhoL * (sL - uL)
        c_R = rhoR * (sR - uR)

        u_star = (-c_L * uL + c_R * uR - (pR - pL)) / (-c_L + c_R)
        p_star = pL + c_L * (u_star - uL)
        rhoL_star = rhoL * (sL - uL) / (sL - u_star)
        rhoR_star = rhoR * (sR - uR) / (sR - u_star)
        EL_star = ((sL - uL) * EL - pL * uL + p_star * u_star) / (sL - u_star)
        ER_star = ((sR - uR) * ER - pR * uR + p_star * u_star) / (sR - u_star)

        if sL > 0:
            E[j] = f_L
        elif sL <= 0 and u_star >= 0:
            E[j] = np.array([
                rhouL - sL * (rhoL_star * u_star - rhouL),
                (rhouL * uL + pL - sL * (rhoL_star * u_star**2 + p_star - (rhouL * uL + pL))) / (1.0 - sL),
                (EL + pL) * uL - sL * ((EL_star + p_star) * u_star - (EL + pL) * uL) / (1.0 - sL)
            ])
        elif u_star < 0 and sR >= 0:
            E[j] = np.array([
                rhouR - sR * (rhoR_star * u_star - rhouR),
                (rhouR * uR + pR - sR * (rhoR_star * u_star**2 + p_star - (rhouR * uR + pR))) / (1.0 - sR),
                (ER + pR) * uR - sR * ((ER_star + p_star) * u_star - (ER + pR) * uR) / (1.0 - sR)
            ])
        elif sR < 0:
            E[j] = f_R

    return E

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
        
        AQ = R @ Lambda @ Rinv @ dQ # matrix multiplication
        
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

def compute_jacobian(Q):
    A_p = np.zeros((jmax,3,3))
    A_n = np.zeros((jmax,3,3))
    A   = np.zeros((jmax,3,3))
    sigma = np.zeros(jmax)
    for j in range(0,jmax):
        rho = Q[j,0]
        velo_x = Q[j,1]/Q[j,0]
        press = (gamma-1.0)*(Q[j,2]-Q[j,1]**2/2.0/Q[j,0])
        if(press<0):
            print("pressure is negative! at ",j,Q[j])
            sys.exit()
        total_h = (Q[j,2]+press)/rho
        acoustic = np.sqrt(gamma*press/rho)
        # print(acoustic)
        b1 = 0.5*velo_x**2*(gamma-1.0)/acoustic**2
        b2 = (gamma-1.0)/acoustic**2
        R = np.array([[1.0,1.0,1.0],
                     [velo_x-acoustic,        velo_x,        velo_x+acoustic],
                     [total_h-velo_x*acoustic, 0.5*velo_x**2, total_h+velo_x*acoustic]])
        R_inv = np.array([[0.5*(b1+velo_x/acoustic), -0.5*(1/acoustic+b2*velo_x), 0.5*b2],
                         [1.0-b1,                     b2*velo_x,                  -b2],
                         [0.5*(b1-velo_x/acoustic),  -0.5*(b2*velo_x-1/acoustic), 0.5*b2]])
        lambda_p = np.array([[0.5*(velo_x-acoustic+abs(velo_x-acoustic)), 0.0,                     0.0],
                             [0.0,                                        0.5*(velo_x+abs(velo_x)),0.0],
                             [0.0,                                        0.0                     ,0.5*(velo_x+acoustic+abs(velo_x+acoustic))]])
        lambda_n = np.array([[0.5*(velo_x-acoustic-abs(velo_x-acoustic)), 0.0,                     0.0],
                             [0.0,                                        0.5*(velo_x-abs(velo_x)),0.0],
                             [0.0,                                        0.0                     ,0.5*(velo_x+acoustic-abs(velo_x+acoustic))]])
        lambda_t = np.array([[velo_x-acoustic,0,0],
                             [0,velo_x,0],
                             [0,0,velo_x+acoustic]])
        A_p[j]   = R @ lambda_p @ R_inv
        A_n[j]   = R @ lambda_n @ R_inv
        A[j]     = A_p[j] + A_n[j]
        sigma[j] = abs(velo_x) + acoustic

    return A, A_p, A_n, sigma

def Roe_FDS(Q, order, kappa, nmax, print_interval, time_integration, T_order):
    E = np.zeros([jmax, 3])
    results = []
    
    if time_integration == 0:  # explicit
        for n in range(nmax):
            if n % print_interval == 0:
                print(f'n = {n : 4d} : CFL = {calc_CFL(Q) : .4f}')
                results.append(Q.copy())

            Qold = Q.copy()
            
            coefs = [0.5, 1.0]
            for coef in coefs:
                QL, QR = MUSCL(Qold, order, kappa)
                Roe_flux(QL, QR, E)
                # HLLC(QL, QR, E)
                
                for j in range(1, jmax - 1):
                    Qold[j] = Q[j] - coef * dtdx * (E[j] - E[j-1])
                
                Qold[0] = Q[0]
                Qold[-1] = Q[-1]
            
            Q[:] = Qold[:]
    
    else:  # implicit
        internal_itr = 30
        for n in range(nmax):
            if n % print_interval == 0:
                print(f'n = {n : 4d} : CFL = {calc_CFL(Q) : .4f}')
                results.append(Q.copy())
            
            Qold = Q.copy()
            Q_m = Q.copy()
            Q_m_cons = Q.copy()
            if(n==0):
                Qold2 = Q.copy()
            
            dq_max = 1.0
            dq = np.zeros([jmax, 3])
            if(T_order == 1):
                for m in range(internal_itr):
                    QL, QR = MUSCL(Q_m, order, kappa)
                    Roe_flux(QL, QR, E)
                    # HLLC(QL, QR, E)
                    
                    A, A_p, A_n, sigma = compute_jacobian(Q_m_cons)
                    
                    # 1st sweep
                    for j in range(1, jmax-1):
                        dq[j] = (-(Q_m_cons[j] - Qold[j]) - dtdx * (E[j] - E[j-1]) + dtdx * np.dot(A_p[j-1], dq[j-1])) / (1.0 + dtdx * sigma[j])
                    dq[0] = dq[1]#(-(Q_m_cons[0] - Qold[0]) - dtdx * (E[0] - E[0]) + dtdx * np.dot(A_p[0], dq[0])) / (1.0 + dtdx * sigma[0])
                    dq[-1] = dq[-2]#(-(Q_m_cons[-1] - Qold[-1]) - dtdx * (E[-1] - E[-1]) + dtdx * np.dot(A_p[-1], dq[-1])) / (1.0 + dtdx * sigma[-1])
                    
                    # 2nd sweep
                    for j in range(jmax-2, 0, -1):
                        dq[j] = dq[j] - dtdx * np.dot(A_n[j+1], dq[j+1]) / (1.0 + dtdx * sigma[j])
                    dq[jmax-1] = dq[-2] #dq[jmax-1] - dtdx * np.dot(A_n[jmax-1], dq[jmax-1]) / (1.0 + dtdx * sigma[jmax-1])
                    dq[0] = dq[1] #dq[0] - dtdx * np.dot(A_n[0], dq[0]) / (1.0 + dtdx * sigma[0])

                    Q_m_cons[:] = Q_m_cons[:] + dq[:]
                    Q_m_cons[0] = Q_m_cons[1]
                    Q_m_cons[-1] = Q_m_cons[-2]
                    Q_m[:] = Q_m_cons[:]
                    
                    dq_max = np.max(dq)
                
                    if (dq_max < 1.e-2):
                        print(f'Iteration {m}, dq_max: {dq_max}')
                        break
            elif(T_order == 2):
                for m in range(internal_itr):
                    QL, QR = MUSCL(Q_m, order, kappa)
                    Roe_flux(QL, QR, E)
                    # HLLC(QL, QR, E)
                    
                    A, A_p, A_n, sigma = compute_jacobian(Q_m_cons)
                    
                    # 1st sweep
                    for j in range(1, jmax-1):
                        dq[j] = (-(3.0*Q_m_cons[j] - 4.0*Qold[j]+Qold2[j])/3.0 - 2.0*dtdx * (E[j] - E[j-1])/3.0 + 2.0*dtdx * np.dot(A_p[j-1], dq[j-1])/3.0) / (1.0 + 2.0*dtdx * sigma[j]/3.0)
                    # dq[0] = (-(3.0*Q_m_cons[0] - 4.0*Qold[0]+Qold2[j])/3.0 - 2.0*dtdx * (E[0] - E[0])/3.0 + 2.0*dtdx * np.dot(A_p[0], dq[0])/3.0) / (1.0 + 2.0*dtdx * sigma[0]/3.0)
                    dq[0] = dq[1]
                    dq[-1] = dq[-2]
                    # dq[-1] = (-(3.0*Q_m_cons[-1] - 4.0*Qold[-1]+Qold2[j])/3.0 - 2.0*dtdx * (E[-1] - E[-1])/3.0 + 2.0*dtdx * np.dot(A_p[-1], dq[-1])/3.0) / (1.0 + 2.0*dtdx * sigma[-1]/3.0)
                    
                    # 2nd sweep
                    for j in range(jmax-2, 0, -1):
                        dq[j] = dq[j] - 2.0*dtdx * np.dot(A_n[j+1], dq[j+1])/3.0 / (1.0 + 2.0*dtdx * sigma[j]/3.0)
                    dq[0] = dq[1]
                    dq[-1] = dq[-2]
                    # dq[jmax-1] = dq[jmax-1] - 2.0*dtdx * np.dot(A_n[jmax-1], dq[jmax-1])/3.0 / (1.0 + 2.0*dtdx * sigma[jmax-1]/3.0)
                    # dq[0] = dq[0] - 2.0*dtdx * np.dot(A_n[0], dq[0])/3.0 / (1.0 + 2.0*dtdx * sigma[0]/3.0)
                    Q_m_cons[:] = Q_m_cons[:] + dq[:]
                    Q_m_cons[0] = Q_m_cons[1]
                    Q_m_cons[-1] = Q_m_cons[-2]
                    Q_m[:] = Q_m_cons[:]
                    
                    dq_max = np.max(dq)
                    if (dq_max < 1.e-5):
                        print(f'Iteration {m}, dq_max: {dq_max}')
                        break
            # Neumann BC
            Q_m_cons[0] = Q_m_cons[1]
            Q_m_cons[-1] = Q_m_cons[-2]

            Qold2[:] = Qold[:]
            Qold[:] = Q_m_cons[:]
            Q[:] = Q_m_cons[:]
            print("step done")
    return results

def update_plot(frame, x, line):
    line.set_ydata(frame[:, 0])
    return line,

if __name__ == "__main__":
    nmax = 20
    print_interval = 1

    order = 3

    kappa = 1/3
    time_integration = 1
    time_order = 2

    # Q1 = init_Sod()
    Q = init_Sod()
    Q2 = init_Sod()
    # Q1 = init_Shu_Osher()
    # Q = init_Shu_Osher()
    # Q2 = init_Shu_Osher()
    # results_exp = Roe_FDS(Q1, order, kappa, nmax, print_interval, 0, time_order)
    results = Roe_FDS(Q, order, kappa, nmax, print_interval, time_integration, time_order)
    results2 = Roe_FDS(Q2, order, kappa, nmax, print_interval, time_integration, 1)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
    plt.rcParams["font.size"] = 22
    ax.set_xlabel('x')
    ax.set_ylabel(r'$\rho$')
    ax.grid(color='black', linestyle='dotted', linewidth=0.5)
    # line, = ax.plot(x, results_exp[99][:, 0], color='black', linewidth=1.5,label='RK2nd')
    line, = ax.plot(x, results[19][:, 0], color='red', linewidth=1.5,label='LU-SGS 2nd')
    line, = ax.plot(x, results2[19][:, 0], color='blue', linewidth=1.5,label = 'LU-SGS 1st')

    # ani = animation.FuncAnimation(
    #     fig, update_plot, frames=results, fargs=(x, line), blit=True, interval=100
    # )
    # ani = animation.FuncAnimation(
    #     fig, update_plot, frames=results2, fargs=(x, line), blit=True, interval=10
    # )
    ax.legend(fontsize='small')
    plt.show()
                                  
                                  