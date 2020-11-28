import numpy as np
import matplotlib.pyplot as plt
import math

def hh_euler(I, tmax, v, mi, hi, ni):
    """
    simulates the Hodgkin-Huxley model of an action potential
    
    I - stimulus current in mA
    tmax - max time value in ms
    v - initial membrane potential
    mi - initial value for m, the fraction of open sodium m-gates
    hi - initial value for h, the fraction of open sodium h-gates
    ni - initial value for n, the fraction of open potassium n-gates 
    """
    
    dt = 0.001
    iterations = math.ceil(tmax/dt)

    g_Na = 120
    e_Na = 115
    g_K = 36
    e_K = -12
    g_L = 0.3
    e_L = 10.6

    # Initializing variable arrays
    t = np.linspace(0,tmax,num=iterations)
    V = np.zeros((iterations,1))
    m = np.zeros((iterations,1))
    h = np.zeros((iterations,1))
    n = np.zeros((iterations,1))
  
    V[0]=v
    m[0]=mi
    h[0]=hi
    n[0]=ni

    #Euler method
    for i in range(iterations-1):
        V[i+1] = V[i] + dt*(g_Na*m[i]**3*h[i]*(e_Na-(V[i]+65)) + g_K*n[i]**4*(e_K-(V[i]+65)) + \
            g_L*(e_L-(V[i]+65)) + I)
        m[i+1] = m[i] + dt*(a_M(V[i])*(1-m[i]) - b_M(V[i])*m[i])
        h[i+1] = h[i] + dt*(a_H(V[i])*(1-h[i]) - b_H(V[i])*h[i])
        n[i+1] = n[i] + dt*(a_N(V[i])*(1-n[i]) - b_N(V[i])*n[i])

    plt.plot(t, V)
    plt.xlabel('Time')
    plt.ylabel('Membrane Potential')
    plt.title('Membrane Potential vs. Time')
    plt.show()

# alpha and beta functions for the gating variables 
def a_M(V):
    return (2.5-0.1*(V+65)) / (np.exp(2.5-0.1*(V+65)) -1)

def b_M(V):  
    return 4*np.exp(-(V+65)/18)

def a_H(V):
    return 0.07*np.exp(-(V+65)/20)

def b_H(V):
    return 1./(np.exp(3.0-0.1*(V+65))+1)

def a_N(V):
    return (0.1-0.01*(V+65)) / (np.exp(1-0.1*(V+65)) -1)

def b_N(V):
    return 0.125*np.exp(-(V+65)/80)

hh_euler(0.08, 40, -65, 0.4, 0.2, 0.5)