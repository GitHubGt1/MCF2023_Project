import scipy.optimize as optimize
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
import pandas as pd

sys.path.append('/home/greta/Desktop/uni/Computazionali/MCF2023_Project/Modulo.py')
sys

import Modulo as md
    
####################################################################################################
#Import dei dati dal file csv

data = pd.read_csv('/home/greta/Desktop/uni/Computazionali/MCF2023_Project/observed_starX.csv')
data

theta = np.radians(45)

l_emesse = data['lambda (nm)'].values
photons = data['photons'].values

plt.bar(l_emesse, photons, width = l_emesse[1] - l_emesse[0],  color = 'indianred', label = 'dati')
plt.xlabel("Lunghezza d'onda (nm)")
plt.ylabel('Numero di fotoni osservati')
plt.title('Dati', color = 'slategrey')
plt.legend()
plt.show()


filtro = photons != 0
l_emesse = l_emesse[filtro]
photons = photons[filtro]

sigma_y = np.sqrt(photons)

###################################################################################################
#Fit con la distribuzione adattata con lo scattering di Rayleigh
p0 = np.array([8*10**3, theta, 300], dtype = 'float64')
params, params_covariance = optimize.curve_fit(md.N_obs_opt, l_emesse*10**(-9), photons, p0 = p0, sigma = sigma_y, absolute_sigma = True)
p_err = np.sqrt(np.diag(params_covariance))
p_names = ['Temperatura', 'Angolo zenitale', 'Costante di scala']
params[1] = np.degrees(params[1])
p_err[1] = np.degrees(p_err[1])
print('----------------------------------------------')
print('Parametri ottenuti dal fit:\n')
for p, pn ,pe, u in zip(params, p_names, p_err, ['K', '°', '']):
    print('{:} = ({:.3} ± {:.3}) {:}\n'.format(pn, abs(p), abs(pe), u))

########################################################################################
#Calcolo del chi^2
y_fit = md.N_obs_opt(l_emesse*10**(-9), params[0], np.radians(params[1]), params[2])
df = len(photons)-2
squarederror1 = np.power((y_fit - photons) / sigma_y , 2)
chi2 = squarederror1.sum()
p = stats.chi2.sf(chi2, df)

print('\n------------------------------------------------------------------------')

print('Chi^2 =  {:6.5f}'.format(chi2))
print('Chi^2 ridotto = {:6.5f}'.format(chi2/df))
print('Probabilità di accordo P = {:.4} %'.format(p*100))

###########################################################################################
#PLot del fit
fig = plt.figure(figsize=(10,8))
plt.bar(l_emesse, photons, width = l_emesse[1] - l_emesse[0],  color = 'indianred', label = 'Dati')
plt.plot(l_emesse, y_fit, color = 'navy', label = 'Fit con parametri:\nT={:.3}\nθ={:.3}\nA={:.3}'.format(abs(params[0]), abs(params[1]), abs(params[2])))
plt.fill_between(l_emesse, md.N_obs_opt(l_emesse*10**(-9), params[0]+p_err[0], np.radians(params[1]+p_err[1]), params[2]+p_err[2]), md.N_obs_opt(l_emesse*10**(-9), params[0]-p_err[0], np.radians(params[1]-p_err[1]), params[2]-p_err[2]), color = 'lightblue', alpha = 0.5, label = 'Banda di variazione del fit\nconsiderando gli errori\nsu parametri')
plt.xlabel("Lunghezza d'onda (nm)")
plt.ylabel('Numero fotoni')
plt.title('Fit dei dati con la distribuzione dei fotoni osservati\nconsiderando lo scattering di Rayleigh\n'+r'$N_{obs}(\lambda) = \frac{A}{max(N_{obs})}\cdot D(\lambda, T)\cdot e^{-\beta(\lambda) S(\theta)}$', color = 'slategrey')
plt.legend(loc = 'upper right')
plt.text(x = 2000, y = 150, s ='Temperatura della StellaX:\n{:.0f} ± {:.0f} K'.format(params[0], p_err[0]), fontsize = 'large', horizontalalignment='center', color = 'slategrey', backgroundcolor = 'lightblue')
plt.show()


