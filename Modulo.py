import numpy as np
import scipy.optimize as optimize

h = 6.62618*10**(-34) #costante di Plank in Js
c = 3*10**8 #velocità della luce in m/s
kb = 1.38006*10**(-23) #costante di Boltzmann in J/K

n = 1.00029
N = 2.504 * 10**25 #molecole/m^3
R_t = 6.371*10**6 #m
S_z = 8000 #m

def corpo_nero(l, T):
    """
        Densità di fotoni per lunghezza d'onda l, emessi da un corpo nero a temperatura T.
    """
    c1 = 2*h*c**2/(l**5)
    esp = h*c/(l*kb*T)
    c2 = l/(h*c)
    return c1* 1/(np.exp(esp)-1)*c2

def corpo_nero_opp(l,T):
    return -corpo_nero(l,T)


def corpo_nero_norm(lamb,T,n):
        """
        Densità di fotoni per lunghezza d'onda lamb, emessi da un corpo nero a temperatura T.
        Unita di misura Jm^(-3)s^(-1)
    
        """
        return corpo_nero(lamb,T)/n

def coeff_rayleigh(l, n, N):
    """
        Coefficiente che descrive lo scattering di Rayleigh dei fotoni da parte di un materiale
        costituito da molecole (centri di diffusione) con dimensioni molto minori della lunghezza d'onda.
    """
    c1 = 8*np.pi**3/(3*np.power(l,4)*N)
    return c1 * (n**2-1)**2

def massa_d_aria(theta, S_z, R):
    """
        Funzione che restituisce lo spessore di una massa d'aria ad un angolo theta dallo zenith,
        a partire da R che rappresenta il raggio della Terra e S_z ovvero lo spessore dell'atmosfera allo zenith.
    """
    rad = np.power(R* np.cos(theta),2)+2*R*S_z+ S_z**2
    return np.sqrt(rad) - R*np.cos(theta)

def N_obs(l,t, theta):
    """
    Funzione che descrive il numero di fotoni osservati in funzione della lunghezza d'onda
    Tenendo conto dello scattering di Rayleigh

    l: lunghezze d'onda
    n_i: numero iniziale dei fotoni alla lunghezza l
    theta: angolo zenitale della sorgente
    """
    return corpo_nero(l,t)*np.exp(-coeff_rayleigh(l,n,N)*massa_d_aria(theta,S_z,R_t))

def N_obs_opp(l, t ,theta):
     return -N_obs(l, t, theta)

def N_obs_norm(l, t, theta,n):
    return N_obs(l, t, theta)/n

def N_obs_inv(l, n_obs, theta):
     n_obs*np.exp(coeff_rayleigh(l,n,N)*massa_d_aria(theta,S_z,R_t))

def N_obs_opt(l, t, theta, A):
    X_max = optimize.minimize(corpo_nero_opp, x0=400*10**(-9), args=(t))        
    n = corpo_nero(X_max.x[0], t) 
    return A*N_obs(l, t, theta)/n









