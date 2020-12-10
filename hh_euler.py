import numpy as np
import matplotlib.pyplot as plt
import math

def hh_euler(I, tmax, v, m_0, h_0, n_0):
    """
    simulates the Hodgkin-Huxley model of an action potential
    
    I - stimulus current in mA
    tmax - max time value in ms
    v - initial membrane potential
    m_0 - initial value for m, the fraction of open sodium m-gates
    h_0 - initial value for h, the fraction of open sodium h-gates
    n_0 - initial value for n, the fraction of open potassium n-gates 
    """
    
    dt = 0.001
    iterations = math.ceil(tmax/dt)

    g_Na = 120
    v_Na = 115
    g_K = 36
    v_K = -12
    g_L = 0.3
    v_L = 10.6

    # Initializing variable arrays
    t = np.linspace(0,tmax,num=iterations)
    V = np.zeros((iterations,1))
    m = np.zeros((iterations,1))
    h = np.zeros((iterations,1))
    n = np.zeros((iterations,1))
  
    V[0]=v
    m[0]=m_0
    h[0]=h_0
    n[0]=n_0

    # Euler's method
    for i in range(iterations-1):
        curr_v = V[i] + 65
        V[i+1] = V[i] + dt*(g_Na*m[i]**3*h[i]*(v_Na-(curr_v)) + g_K*n[i]**4*(v_K-(curr_v)) + \
            g_L*(v_L-(curr_v)) + I)
        m[i+1] = m[i] + dt*(a_M(curr_v)*(1-m[i]) - b_M(curr_v)*m[i])
        h[i+1] = h[i] + dt*(a_H(curr_v)*(1-h[i]) - b_H(curr_v)*h[i])
        n[i+1] = n[i] + dt*(a_N(curr_v)*(1-n[i]) - b_N(curr_v)*n[i])
    return t, V

# alpha and beta functions for the gating variables 
def a_M(v):
    return 0.1*(25 - v)/(np.exp((25-v)/10) - 1)

def b_M(v):
    return 4*np.exp(-1*v/18)

def a_H(v):
    return 0.07*np.exp(-1*v/20)

def b_H(v):
    return 1/(np.exp((30-v)/10) + 1)

def a_N(v):
    return 0.01 * (10 - v)/(np.exp((10-v)/10) - 1)

def b_N(v):
    return 0.125*np.exp(-1*v/80)


if __name__ == "__main__":
    I_0 = [1, 5, 10, 25]
    sub_pos = [(0,0),(0,1),(1,0),(1,1)]
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    for i in range(4):
        t, V = hh_euler(I_0[i], 100, -65, 0.4, 0.2, 0.5)
        pos = sub_pos[i]
        axs[pos].plot(t, V)
        axs[pos].set(xlabel='t (ms)', ylabel='V (mV)')
        axs[pos].set_title(str(I_0[i]) + ' mV Stimulus Current')
        axs[pos].set_ylim([-80, 40])
    plt.suptitle('Membrane Potential vs. Time for Varying Stimulus Currents', fontsize=18)
    plt.show()