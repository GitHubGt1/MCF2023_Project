import numpy as np
import matplotlib.pyplot as plt
import scipy
import matplotlib.colors as colors
import sys
import math
import argparse
import numpy.random as random
import scipy.integrate as integrate

sys.path.append('/home/greta/Desktop/uni/Computazionali/Progetto/Modulo.py')
sys

import Modulo as md

def wavelength_to_rgb(wavelength, gamma=0.8):
    ''' taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
    This converts a given wavelength of light to an 
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    Additionally alpha value set to 0.5 outside range
    '''
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 750:
        A = 1.
    else:
        A=0.5
    if wavelength < 380:
        wavelength = 380.
    if wavelength >750:
        wavelength = 750.
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return (R,G,B,A)


####################################################################################
#Definizione colormap per mappare le lunghezze d'onda in colori
clim=(380,750)
norm = plt.Normalize(*clim)
wl = np.arange(clim[0],clim[1]+1,2)
colorlist = list(zip(norm(wl),[wavelength_to_rgb(w) for w in wl]))
spectralmap = colors.LinearSegmentedColormap.from_list("spectrum", colorlist)

####################################################################################
#Temperature: Sole, Antares, Sirius, Rigel
S = ['Sun', 'Antares', 'Sirius', 'Rigel']
T = [5.75*10**3, 3*10**3, 10*10**3, 25*10**3] #K


####################################################################################
#Partendo da 1000 fotoni con lunghezza d'onda uniformemente distribuita nel visibile, 
#li filtro secondo la distriubuzione del corpo nero riscalata ad 1 con il metodo hit or miss

n_sample = 10000

l = random.uniform(low = 380, high = 750, size = n_sample)

stelle = {}

for s,t in zip(S,T):
    valore=500                                                        

    X_max = scipy.optimize.minimize(md.corpo_nero_opp, x0=valore*10**(-9), args=(t))        
    norm = md.corpo_nero(X_max.x[0], t) 

    N0 = md.corpo_nero_norm(l*10**(-9), t, norm)
    f = random.uniform(low = 0, high = 1, size = n_sample)               
    mask = f <= N0
    l_emesse = l[mask]
    f_emessi = N0[mask]
    ind = np.argsort(l_emesse)
    l_emesse = l_emesse[ind]
    f_emessi = f_emessi[ind]
    stelle.update({s : [l_emesse, f_emessi, t]})

######################################################################################
#Definizione delle opzioni per runnare il programma
def parse_arguments():
    
    parser = argparse.ArgumentParser(description = "Specificare la stella di cui si vuole effettuare l'analisi",
                                     usage      ='python3 Tramonti.py  --opzione')
    parser.add_argument('-m', '--main',    action='store_true',   help="Mostra lo studio di un flusso di fotoni solari tenendo conto dello scattering di Rayleigh")
    parser.add_argument('-a', '--antares',    action='store_true',   help='Mostra lo studio di un flusso di fotoni prodotti da Sirius tenendo conto dello scattering di Rayleigh')
    parser.add_argument('-s', '--sirius',    action='store_true',   help='Mostra lo studio di un flusso di fotoni prodotti da Sirius tenendo conto dello scattering di Rayleigh')
    parser.add_argument('-r', '--rigel',    action='store_true',   help='Mostra lo studio di un flusso di fotoni prodotti da Rigel tenendo conto dello scattering di Rayleigh')


    return  parser.parse_args()

def main():

    args = parse_arguments()
    
    if args.main == False and args.antares == False and args.sirius == False and args.rigel == False:
        print('Specificare la stella sorgente del flusso di fotoni da analizzare.\nPer ulteriori informazioni: python3 Tramonti.py --help')

    if args.main == True:
        print('TRAMONTO DEL SOLE')

        ########################################################################
        #Generazione fotoni con il metodo hit or miss

        #Plot dei fotoni generati vs quelli emessi con il metodo hit or miss
        plt.hist(l, bins = 100,  range=(380, 750), label='generati', color = 'rosybrown')
        n, bins1, p = plt.hist(stelle.get('Sun')[0], bins = 100, range=(380, 750), label='selezionate', color = 'teal', alpha = 0.5)
        bincenters = (bins1[:-1]+bins1[1:])/2
        plt.xlabel("Lunghezza d'onda (m)")
        plt.ylabel(r'Densità di fotoni osservati $(\frac{fotoni}{s \cdot m^2})$')
        plt.legend()
        plt.title('Metodo hit or miss per la selezione di samples\nsecondo lo spettro del corpo nero', color = 'slategrey')
        plt.show()

        ####################################################################
        #Spettro di emissione

        #Plot dello spettro di emissione del sole usando la funzione corpo_nero riscalata ad 1
        f_emessi = stelle.get('Sun')[1]
        l_emesse = stelle.get('Sun')[0]
        t = stelle.get('Sun')[2]
        y = f_emessi
        X, Y =  np.meshgrid(l_emesse, y)
        extent = (np.min(l_emesse), np.max(l_emesse), np.min(y), np.max(y)+0.1)
        
        
        #calcolo della lunghezza d'onda per cui si ha la massima emissione
        i = np.where(f_emessi == np.max(f_emessi))[0]
        l_max = l_emesse[i[0]]

        fig, axs = plt.subplots(1, 1, figsize=(6,5), tight_layout=True)
        plt.plot(l_emesse, f_emessi, '.', color='darkred')
        plt.axvline(x = l_max, label = 'λ_max = {:.5} nm'.format(l_max), color = 'slategrey')
        plt.imshow(X, clim=clim, extent=extent, cmap=spectralmap, aspect='auto')
        plt.xlabel("Lunghezza d'onda (nm)")
        plt.ylabel(r"Densità di fotoni emessi \nper lunghezza d'onda $(\frac{fotoni}{m^3})$")
        plt.title('Spettro di emissione del sole (Riscalato ad 1)', color = 'slategrey')
        plt.fill_between(l_emesse, f_emessi, np.full(len(l_emesse), np.max(y)+0.1), color='w')
        plt.legend()
        plt.show()

        ###############################################################################################7
        #Scattering di Rayleigh con sorgente allo zenith

        #Distribuzione dei fotoni considerando lo scattering di Rayleigh con il sole allo Zenith
        theta = math.radians(0)
        valore=500*10**(-9)                                                                          

        X_max = scipy.optimize.minimize(md.N_obs_opp, x0=valore, args=(t, theta))        
        norm_z = md.N_obs(X_max.x[0], t, theta)     

        l0 = random.uniform(low = 380, high = 750, size = n_sample)
        N_s_z_base = md.N_obs_norm(l0*10**(-9), t, theta, norm_z)
        value = random.uniform(low = 0, high = np.max(N_s_z_base), size = n_sample)
        mask = value <= N_s_z_base
        l_scatter = l0[mask]
        l_scatter = np.sort(l_scatter)

        plt.title('Metodo hit or miss per la selezione di samples\n considerando lo scattering di Rayleigh con il sole allo Zenith', color = 'slategrey')
        n0, b, p = plt.hist(l0, bins = 100, color = 'navajowhite', label = 'Generati')
        n2, bins2, p = plt.hist(l_scatter, bins = 100, color = 'tomato', label = 'Selezionati', alpha = 0.5)
        bincenters2 = (bins2[:-1] + bins2[1:])/2
        plt.xlabel("Lunghezza d'onda (nm)")
        plt.ylabel(r'Numero di fotoni per unità di superficie e tempo  $(\frac{fotoni}{s\cdot m^2})$')
        plt.yscale('log')
        plt.legend()
        plt.show()

        ###############################################################################################7
        #Scattering di Rayleigh con sorgente all'orizzonte

        #Distribuzione dei fotoni considerando lo scattering di Rayleigh con il sole all'orizzonte
        theta = math.radians(90)
        valore=500*10**(-9)                                                                          

        X_max = scipy.optimize.minimize(md.N_obs_opp, x0=valore, args=(t, theta))        
        norm_o = md.N_obs(X_max.x[0], t, theta)     

        l2 = random.uniform(low = 380, high = 750, size = n_sample)
        N_s_o_base = md.N_obs_norm(l2*10**(-9), t, theta, norm_o)
        value = random.uniform(low = 0, high = np.max(N_s_o_base), size = n_sample)
        mask = value <= N_s_o_base
        l_scatter_o = l2[mask]
        l_scatter_o = np.sort(l_scatter_o)


        plt.title("Metodo hit or miss per la selezione di samples\n considerando lo scattering di Rayleigh con il sole all'orizzonte", color = 'slategrey')
        n1, bins, p = plt.hist(l2, bins = 100, color = 'navajowhite', label = 'Generati')
        n3, bins3, p = plt.hist(l_scatter_o, bins = 100, color = 'tomato', label = "Selezionati", alpha = 0.5)
        bincenters3 = (bins3[:-1] + bins3[1:])/2
        plt.xlabel("Lunghezza d'onda (nm)")
        plt.ylabel(r'Numero di fotoni per unità di superficie e tempo  $(\frac{fotoni}{s\cdot m^2})$')
        plt.yscale('log')
        plt.legend()
        plt.show()

        #########################################################################################################
        #Confronto distribuzioni teoriche ed effettive

        #Calcolo gli integrali per normalizzare la distribuzione ed avere significato di probabilità
        x = np.linspace(380*10**(-9),750*10**(-9), 10000)
        y_1 = md.corpo_nero(x, t)
        norm1 = integrate.simpson(y_1,x)
        y_2 = md.N_obs(x, t, 0)
        norm2 = integrate.simpson(y_2, x)
        y_3 = md.N_obs(x, t, np.pi/2)
        norm3 = integrate.simpson(y_3, x)

        #Calcolo il numero di fotoni aspettati moltiplicando la probabilità di ogni bin per il numero di fotoni totali
        #Probabilità associata ai bin: (larghezza del bin)*(valore della distribuzione nel bincenter)
        base1 = (bincenters[1]-bincenters[0])*10**(-9)
        base2 = (bincenters2[1]-bincenters2[0])*10**(-9)
        base3 = (bincenters3[1]-bincenters3[0])*10**(-9)
        area_1 = base1*md.corpo_nero_norm(bincenters*10**(-9), t, norm1)                            
        area_2 = base2*md.N_obs_norm(bincenters2*10**(-9), t, 0, norm2)
        area_3 = base3*md.N_obs_norm(bincenters3*10**(-9), t, math.radians(90), norm3)
        dist_1 = area_1*len(l_emesse)
        dist_2 = area_2*len(l_scatter)
        dist_3 = area_3*len(l_scatter_o)

        #Confronto grafico fotoni osservati-fotoni aspettati
        fig, ((ax1), (ax2), (ax3)) = plt.subplots(1, 3, figsize = (9,8), sharey = True)

        fig.suptitle('Confronto distribuzioni teoriche ed effettive', color = 'slategrey')
        ax1.hist(l_emesse, bins = 100, color = 'seagreen', label = 'Osservati')
        ax1.plot(bincenters, dist_1, color = 'darkgreen', label = 'Aspettati')
        ax1.set_title('Distribuzione del corpo nero', color = 'slategrey')
        ax2.hist(l_scatter, bins = 100, color = 'skyblue', label = 'Osservati')
        ax2.plot(bincenters2, dist_2, color = 'navy', label = 'Aspettati')
        ax2.set_title('Distribuzione scattering\ncon sorgente allo Zenith', color = 'slategrey')
        ax3.hist(l_scatter_o, bins = 100, color = 'indianred', label = 'Osservati')
        ax3.plot(bincenters3, dist_3, color = 'maroon', label = 'Aspettati')
        ax3.set_title("Distribuzione scattering\ncon sorgente all'orizzonte", color = 'slategrey')
        ax1.set_ylabel(r'Numero di fotoni per unità di superficie e tempo  $(\frac{fotoni}{s\cdot m^2})$')
        for ax in fig.get_axes():
            ax.set_xlabel("Lunghezza d'onda (nm)")
            ax.legend()

        plt.show()


        #####################################################################################################
        #Confronto fotoni diffusi per scattering di Rayleigh con il sole allo zenith e all'orizzonte

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (8,9), gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle("Confronto distribuzione dei fotoni emessi e di quelli osservati\n considerando lo scattering di Rayleigh", color = 'slategrey')

        ax1.hist(l_emesse, bins = 100, color = 'navajowhite', label = 'Senza assorbimento')
        ax1.hist(l_scatter, bins = 100, color = 'seagreen', label = 'Scattering di Rayleigh con il sole allo zenith', alpha = 0.5)
        ax1.hist(l_scatter_o, bins = 100, color = 'tomato', label = "Scattering di Rayleigh con il sole all'orizzonte")

        #Plot della differenza tra i 10000 fotoni generati casualmente e quelli selezionati dall'hit or miss, proporzionale alla densità di fotoni diffusi
              
        y2 = n-n3
        y1 = n-n2
        X2, Y2 = np.meshgrid(bincenters3, y2)

        ax2.plot(bincenters3, n-n2, color = 'b', label = 'Scattering con il sole allo zenith')
        ax2.plot(bincenters3, n-n3, color = 'black', label = "Scattering con il sole all'orizzonte")
        ax2.imshow(X2, clim=clim, extent=(np.min(bincenters3), np.max(bincenters3), np.min(y1), np.max(y2)+10), cmap=spectralmap, aspect='auto')
        ax2.fill_between(bincenters3, y2, np.full(len(bincenters3), np.max(y2)+10), color='w')
        ax2.set_title('Andamento proporzionale alla densità di fotoni diffusi', fontsize = 10)
        ax2.set_xlabel("Lunghezza d'onda (nm)")
        ax1.set_ylabel(r'Densità di fotoni osservati $(\frac{fotoni}{s \cdot m^2})$')
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        for ax in fig.get_axes():
            ax.legend(loc = 'lower left')
        plt.show()

        ################################################################################################
        #Studio del flusso integrato in funzione dell'angolo del sole rispetto allo zenith

        theta = np.radians(np.linspace(0, 90, 50))
        valore=500*10**(-9)                                                                          
        simpson_int = np.empty(50)
        for i in range(0, 50):
            X_max = scipy.optimize.minimize(md.N_obs_opp, x0=valore, args=(t, theta[i]))        
            norm_ = md.N_obs(X_max.x[0], t, theta[i])
            y_base = md.N_obs_norm(l_emesse*10**(-9), t, theta[i], norm_)
            value = random.uniform(low = 0, high = np.max(y_base), size = len(l_emesse))
            mask = value <= y_base
            l_int = l_emesse[mask]
            y = y_base[mask]
            ind = np.argsort(l_int)
            l_int = l_int[ind]
            y = y[ind]
            hist = np.histogram(l_int, bins = 100)
            bincenters = (hist[1][:-1] + hist[1][1:])/2
            simpson_int[i] = integrate.simpson(hist[0], bincenters)

        plt.plot(np.degrees(theta), simpson_int, 'o', color = 'indianred')
        plt.xlabel('Angolo del sole rispetto allo zenith (°)')
        plt.ylabel('Flusso integrato di fotoni')
        plt.title("Flusso integrato dei fotoni in funzione\nall'angolo del sole rispetto allo Zenith", color = 'slategrey')
        plt.show()

    if args.antares == True:
        print('TRAMONTO DI Antares')

        ########################################################################
        #Generazione fotoni con il metodo hit or miss

        #Plot dei fotoni generati vs quelli emessi con il metodo hit or miss
        plt.hist(l, bins = 100,  range=(380, 750), label='generati', color = 'rosybrown')
        n, bins1, p = plt.hist(stelle.get('Antares')[0], bins = 100, range=(380, 750), label='selezionate', color = 'teal', alpha = 0.5)
        bincenters = (bins1[:-1]+bins1[1:])/2
        plt.xlabel("Lunghezza d'onda (m)")
        plt.ylabel(r'Densità di fotoni osservati $(\frac{fotoni}{s \cdot m^2})$')
        plt.legend()
        plt.title('Metodo hit or miss per la selezione di samples\nsecondo lo spettro del corpo nero', color = 'slategrey')
        plt.show()

        ####################################################################
        #Spettro di emissione

        #Plot dello spettro di emissione di Antares usando la funzione corpo_nero riscalata ad 1
        f_emessi = stelle.get('Antares')[1]
        l_emesse = stelle.get('Antares')[0]
        t = stelle.get('Antares')[2]
        y = f_emessi
        X, Y =  np.meshgrid(l_emesse, y)
        extent = (np.min(l_emesse), np.max(l_emesse), np.min(y), np.max(y)+0.1)
        
        
        #calcolo della lunghezza d'onda per cui si ha la massima emissione
        i = np.where(f_emessi == np.max(f_emessi))[0]
        l_max = l_emesse[i[0]]

        fig, axs = plt.subplots(1, 1, figsize=(6,5), tight_layout=True)
        plt.plot(l_emesse, f_emessi, '.', color='darkred')
        plt.axvline(x = l_max, label = 'λ_max = {:.5} nm'.format(l_max), color = 'slategrey')
        plt.imshow(X, clim=clim, extent=extent, cmap=spectralmap, aspect='auto')
        plt.xlabel("Lunghezza d'onda (nm)")
        plt.ylabel(r"Densità di fotoni emessi \nper lunghezza d'onda $(\frac{fotoni}{m^3})$")
        plt.title('Spettro di emissione di Antares (Riscalato ad 1)', color = 'slategrey')
        plt.fill_between(l_emesse, f_emessi, np.full(len(l_emesse), np.max(y)+0.1), color='w')
        plt.legend()
        plt.show()

        ###############################################################################################7
        #Scattering di Rayleigh con sorgente allo zenith

        #Distribuzione dei fotoni considerando lo scattering di Rayleigh con Antares allo Zenith
        theta = math.radians(0)
        valore=500*10**(-9)                                                                          

        X_max = scipy.optimize.minimize(md.N_obs_opp, x0=valore, args=(t, theta))        
        norm_z = md.N_obs(X_max.x[0], t, theta)     

        l0 = random.uniform(low = 380, high = 750, size = n_sample)
        N_s_z_base = md.N_obs_norm(l0*10**(-9), t, theta, norm_z)
        value = random.uniform(low = 0, high = np.max(N_s_z_base), size = n_sample)
        mask = value <= N_s_z_base
        l_scatter = l0[mask]
        l_scatter = np.sort(l_scatter)

        plt.title('Metodo hit or miss per la selezione di samples\n considerando lo scattering di Rayleigh con Antares allo Zenith', color = 'slategrey')
        n0, b, p = plt.hist(l0, bins = 100, color = 'navajowhite', label = 'Generati')
        n2, bins2, p = plt.hist(l_scatter, bins = 100, color = 'tomato', label = 'Selezionati', alpha = 0.5)
        bincenters2 = (bins2[:-1] + bins2[1:])/2
        plt.xlabel("Lunghezza d'onda (nm)")
        plt.ylabel(r'Numero di fotoni per unità di superficie e tempo  $(\frac{fotoni}{s\cdot m^2})$')
        plt.yscale('log')
        plt.legend()
        plt.show()

        ###############################################################################################7
        #Scattering di Rayleigh con sorgente all'orizzonte

        #Distribuzione dei fotoni considerando lo scattering di Rayleigh con Antares all'orizzonte
        theta = math.radians(90)
        valore=500*10**(-9)                                                                          

        X_max = scipy.optimize.minimize(md.N_obs_opp, x0=valore, args=(t, theta))        
        norm_o = md.N_obs(X_max.x[0], t, theta)     

        l2 = random.uniform(low = 380, high = 750, size = n_sample)
        N_s_o_base = md.N_obs_norm(l2*10**(-9), t, theta, norm_o)
        value = random.uniform(low = 0, high = np.max(N_s_o_base), size = n_sample)
        mask = value <= N_s_o_base
        l_scatter_o = l2[mask]
        l_scatter_o = np.sort(l_scatter_o)


        plt.title("Metodo hit or miss per la selezione di samples\n considerando lo scattering di Rayleigh con Antares all'orizzonte", color = 'slategrey')
        n1, bins, p = plt.hist(l2, bins = 100, color = 'navajowhite', label = 'Generati')
        n3, bins3, p = plt.hist(l_scatter_o, bins = 100, color = 'tomato', label = "Selezionati", alpha = 0.5)
        bincenters3 = (bins3[:-1] + bins3[1:])/2
        plt.xlabel("Lunghezza d'onda (nm)")
        plt.ylabel(r'Numero di fotoni per unità di superficie e tempo  $(\frac{fotoni}{s\cdot m^2})$')
        plt.yscale('log')
        plt.legend()
        plt.show()

        #########################################################################################################
        #Confronto distribuzioni teoriche ed effettive

        #Calcolo gli integrali per normalizzare la distribuzione ed avere significato di probabilità
        x = np.linspace(380*10**(-9),750*10**(-9), 10000)
        y_1 = md.corpo_nero(x, t)
        norm1 = integrate.simpson(y_1,x)
        y_2 = md.N_obs(x, t, 0)
        norm2 = integrate.simpson(y_2, x)
        y_3 = md.N_obs(x, t, np.pi/2)
        norm3 = integrate.simpson(y_3, x)

        #Calcolo il numero di fotoni aspettati moltiplicando la probabilità di ogni bin per il numero di fotoni totali
        #Probabilità associata ai bin: (larghezza del bin)*(valore della distribuzione nel bincenter)
        base1 = (bincenters[1]-bincenters[0])*10**(-9)
        base2 = (bincenters2[1]-bincenters2[0])*10**(-9)
        base3 = (bincenters3[1]-bincenters3[0])*10**(-9)
        area_1 = base1*md.corpo_nero_norm(bincenters*10**(-9), t, norm1)                            
        area_2 = base2*md.N_obs_norm(bincenters2*10**(-9), t, 0, norm2)
        area_3 = base3*md.N_obs_norm(bincenters3*10**(-9), t, math.radians(90), norm3)
        dist_1 = area_1*len(l_emesse)
        dist_2 = area_2*len(l_scatter)
        dist_3 = area_3*len(l_scatter_o)

        #Confronto grafico fotoni osservati-fotoni aspettati
        fig, ((ax1), (ax2), (ax3)) = plt.subplots(1, 3, figsize = (9,8), sharey = True)

        fig.suptitle('Confronto distribuzioni teoriche ed effettive', color = 'slategrey')
        ax1.hist(l_emesse, bins = 100, color = 'seagreen', label = 'Osservati')
        ax1.plot(bincenters, dist_1, color = 'darkgreen', label = 'Aspettati')
        ax1.set_title('Distribuzione del corpo nero', color = 'slategrey')
        ax2.hist(l_scatter, bins = 100, color = 'skyblue', label = 'Osservati')
        ax2.plot(bincenters2, dist_2, color = 'navy', label = 'Aspettati')
        ax2.set_title('Distribuzione scattering\ncon sorgente allo Zenith', color = 'slategrey')
        ax3.hist(l_scatter_o, bins = 100, color = 'indianred', label = 'Osservati')
        ax3.plot(bincenters3, dist_3, color = 'maroon', label = 'Aspettati')
        ax3.set_title("Distribuzione scattering\ncon sorgente all'orizzonte", color = 'slategrey')
        ax1.set_ylabel(r'Numero di fotoni per unità di superficie e tempo  $(\frac{fotoni}{s\cdot m^2})$')
        for ax in fig.get_axes():
            ax.set_xlabel("Lunghezza d'onda (nm)")
            ax.legend()

        plt.show()


        #####################################################################################################
        #Confronto fotoni diffusi per scattering di Rayleigh con Antares allo zenith e all'orizzonte

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (8,9), gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle("Confronto distribuzione dei fotoni emessi e di quelli osservati\n considerando lo scattering di Rayleigh", color = 'slategrey')

        ax1.hist(l_emesse, bins = 100, color = 'navajowhite', label = 'Senza assorbimento')
        ax1.hist(l_scatter, bins = 100, color = 'seagreen', label = 'Scattering di Rayleigh con Antares allo zenith', alpha = 0.5)
        ax1.hist(l_scatter_o, bins = 100, color = 'tomato', label = "Scattering di Rayleigh con Antares all'orizzonte")

        #Plot della differenza tra i 10000 fotoni generati casualmente e quelli selezionati dall'hit or miss, proporzionale alla densità di fotoni diffusi
              
        y2 = n-n3
        y1 = n-n2
        X2, Y2 = np.meshgrid(bincenters3, y2)

        ax2.plot(bincenters3, n-n2, color = 'b', label = 'Scattering con Antares allo zenith')
        ax2.plot(bincenters3, n-n3, color = 'black', label = "Scattering con Antares all'orizzonte")
        ax2.imshow(X2, clim=clim, extent=(np.min(bincenters3), np.max(bincenters3), np.min(y1), np.max(y2)+10), cmap=spectralmap, aspect='auto')
        ax2.fill_between(bincenters3, y2, np.full(len(bincenters3), np.max(y2)+10), color='w')
        ax2.set_title('Andamento proporzionale alla densità di fotoni diffusi', fontsize = 10)
        ax2.set_xlabel("Lunghezza d'onda (nm)")
        ax1.set_ylabel(r'Densità di fotoni osservati $(\frac{fotoni}{s \cdot m^2})$')
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        for ax in fig.get_axes():
            ax.legend(loc = 'lower left')
        plt.show()


        ################################################################################################
        #Studio del flusso integrato in funzione dell'angolo di Antares rispetto allo zenith

        theta = np.radians(np.linspace(0, 90, 50))
        valore=500*10**(-9)                                                                          
        simpson_int = np.empty(50)
        for i in range(0, 50):
            X_max = scipy.optimize.minimize(md.N_obs_opp, x0=valore, args=(t, theta[i]))        
            norm_ = md.N_obs(X_max.x[0], t, theta[i])
            y_base = md.N_obs_norm(l_emesse*10**(-9), t, theta[i], norm_)
            value = random.uniform(low = 0, high = np.max(y_base), size = len(l_emesse))
            mask = value <= y_base
            l_int = l_emesse[mask]
            y = y_base[mask]
            ind = np.argsort(l_int)
            l_int = l_int[ind]
            y = y[ind]
            hist = np.histogram(l_int, bins = 100)
            bincenters = (hist[1][:-1] + hist[1][1:])/2
            simpson_int[i] = integrate.simpson(hist[0], bincenters)

        plt.plot(np.degrees(theta), simpson_int, 'o', color = 'indianred')
        plt.xlabel('Angolo di Antares rispetto allo zenith (°)')
        plt.ylabel('Flusso integrato di fotoni')
        plt.title("Flusso integrato dei fotoni in funzione\nall'angolo di Antares rispetto allo Zenith", color = 'slategrey')
        plt.show()
      
    if args.sirius == True:
        print('TRAMONTO DI SIRIUS')

        ########################################################################
        #Generazione fotoni con il metodo hit or miss

        #Plot dei fotoni generati vs quelli emessi con il metodo hit or miss
        plt.hist(l, bins = 100,  range=(380, 750), label='generati', color = 'rosybrown')
        n, bins1, p = plt.hist(stelle.get('Sirius')[0], bins = 100, range=(380, 750), label='selezionate', color = 'teal', alpha = 0.5)
        bincenters = (bins1[:-1]+bins1[1:])/2
        plt.xlabel("Lunghezza d'onda (m)")
        plt.ylabel(r'Densità di fotoni osservati $(\frac{fotoni}{s \cdot m^2})$')
        plt.legend()
        plt.title('Metodo hit or miss per la selezione di samples\nsecondo lo spettro del corpo nero', color = 'slategrey')
        plt.show()

        ####################################################################
        #Spettro di emissione

        #Plot dello spettro di emissione di Sirius usando la funzione corpo_nero riscalata ad 1
        f_emessi = stelle.get('Sirius')[1]
        l_emesse = stelle.get('Sirius')[0]
        t = stelle.get('Sirius')[2]
        y = f_emessi
        X, Y =  np.meshgrid(l_emesse, y)
        extent = (np.min(l_emesse), np.max(l_emesse), np.min(y), np.max(y)+0.1)
        
        
        #calcolo della lunghezza d'onda per cui si ha la massima emissione
        i = np.where(f_emessi == np.max(f_emessi))[0]
        l_max = l_emesse[i[0]]

        fig, axs = plt.subplots(1, 1, figsize=(6,5), tight_layout=True)
        plt.plot(l_emesse, f_emessi, '.', color='darkred')
        plt.axvline(x = l_max, label = 'λ_max = {:.5} nm'.format(l_max), color = 'slategrey')
        plt.imshow(X, clim=clim, extent=extent, cmap=spectralmap, aspect='auto')
        plt.xlabel("Lunghezza d'onda (nm)")
        plt.ylabel(r"Densità di fotoni emessi \nper lunghezza d'onda $(\frac{fotoni}{m^3})$")
        plt.title('Spettro di emissione di Sirius (Riscalato ad 1)', color = 'slategrey')
        plt.fill_between(l_emesse, f_emessi, np.full(len(l_emesse), np.max(y)+0.1), color='w')
        plt.legend()
        plt.show()

        ###############################################################################################7
        #Scattering di Rayleigh con sorgente allo zenith

        #Distribuzione dei fotoni considerando lo scattering di Rayleigh con Sirius allo Zenith
        theta = math.radians(0)
        valore=500*10**(-9)                                                                          

        X_max = scipy.optimize.minimize(md.N_obs_opp, x0=valore, args=(t, theta))        
        norm_z = md.N_obs(X_max.x[0], t, theta)     

        l0 = random.uniform(low = 380, high = 750, size = n_sample)
        N_s_z_base = md.N_obs_norm(l0*10**(-9), t, theta, norm_z)
        value = random.uniform(low = 0, high = np.max(N_s_z_base), size = n_sample)
        mask = value <= N_s_z_base
        l_scatter = l0[mask]
        l_scatter = np.sort(l_scatter)

        plt.title('Metodo hit or miss per la selezione di samples\n considerando lo scattering di Rayleigh con Sirius allo Zenith', color = 'slategrey')
        n0, b, p = plt.hist(l0, bins = 100, color = 'navajowhite', label = 'Generati')
        n2, bins2, p = plt.hist(l_scatter, bins = 100, color = 'tomato', label = 'Selezionati', alpha = 0.5)
        bincenters2 = (bins2[:-1] + bins2[1:])/2
        plt.xlabel("Lunghezza d'onda (nm)")
        plt.ylabel(r'Numero di fotoni per unità di superficie e tempo  $(\frac{fotoni}{s\cdot m^2})$')
        plt.yscale('log')
        plt.legend()
        plt.show()

        ###############################################################################################7
        #Scattering di Rayleigh con sorgente all'orizzonte

        #Distribuzione dei fotoni considerando lo scattering di Rayleigh con Sirius all'orizzonte
        theta = math.radians(90)
        valore=500*10**(-9)                                                                          

        X_max = scipy.optimize.minimize(md.N_obs_opp, x0=valore, args=(t, theta))        
        norm_o = md.N_obs(X_max.x[0], t, theta)     

        l2 = random.uniform(low = 380, high = 750, size = n_sample)
        N_s_o_base = md.N_obs_norm(l2*10**(-9), t, theta, norm_o)
        value = random.uniform(low = 0, high = np.max(N_s_o_base), size = n_sample)
        mask = value <= N_s_o_base
        l_scatter_o = l2[mask]
        l_scatter_o = np.sort(l_scatter_o)


        plt.title("Metodo hit or miss per la selezione di samples\n considerando lo scattering di Rayleigh con Sirius all'orizzonte", color = 'slategrey')
        n1, bins, p = plt.hist(l2, bins = 100, color = 'navajowhite', label = 'Generati')
        n3, bins3, p = plt.hist(l_scatter_o, bins = 100, color = 'tomato', label = "Selezionati", alpha = 0.5)
        bincenters3 = (bins3[:-1] + bins3[1:])/2
        plt.xlabel("Lunghezza d'onda (nm)")
        plt.ylabel(r'Numero di fotoni per unità di superficie e tempo  $(\frac{fotoni}{s\cdot m^2})$')
        plt.yscale('log')
        plt.legend()
        plt.show()

        #########################################################################################################
        #Confronto distribuzioni teoriche ed effettive

        #Calcolo gli integrali per normalizzare la distribuzione ed avere significato di probabilità
        x = np.linspace(380*10**(-9),750*10**(-9), 10000)
        y_1 = md.corpo_nero(x, t)
        norm1 = integrate.simpson(y_1,x)
        y_2 = md.N_obs(x, t, 0)
        norm2 = integrate.simpson(y_2, x)
        y_3 = md.N_obs(x, t, np.pi/2)
        norm3 = integrate.simpson(y_3, x)

        #Calcolo il numero di fotoni aspettati moltiplicando la probabilità di ogni bin per il numero di fotoni totali
        #Probabilità associata ai bin: (larghezza del bin)*(valore della distribuzione nel bincenter)
        base1 = (bincenters[1]-bincenters[0])*10**(-9)
        base2 = (bincenters2[1]-bincenters2[0])*10**(-9)
        base3 = (bincenters3[1]-bincenters3[0])*10**(-9)
        area_1 = base1*md.corpo_nero_norm(bincenters*10**(-9), t, norm1)                            
        area_2 = base2*md.N_obs_norm(bincenters2*10**(-9), t, 0, norm2)
        area_3 = base3*md.N_obs_norm(bincenters3*10**(-9), t, math.radians(90), norm3)
        dist_1 = area_1*len(l_emesse)
        dist_2 = area_2*len(l_scatter)
        dist_3 = area_3*len(l_scatter_o)

        #Confronto grafico fotoni osservati-fotoni aspettati
        fig, ((ax1), (ax2), (ax3)) = plt.subplots(1, 3, figsize = (9,8), sharey = True)

        fig.suptitle('Confronto distribuzioni teoriche ed effettive', color = 'slategrey')
        ax1.hist(l_emesse, bins = 100, color = 'seagreen', label = 'Osservati')
        ax1.plot(bincenters, dist_1, color = 'darkgreen', label = 'Aspettati')
        ax1.set_title('Distribuzione del corpo nero', color = 'slategrey')
        ax2.hist(l_scatter, bins = 100, color = 'skyblue', label = 'Osservati')
        ax2.plot(bincenters2, dist_2, color = 'navy', label = 'Aspettati')
        ax2.set_title('Distribuzione scattering\ncon sorgente allo Zenith', color = 'slategrey')
        ax3.hist(l_scatter_o, bins = 100, color = 'indianred', label = 'Osservati')
        ax3.plot(bincenters3, dist_3, color = 'maroon', label = 'Aspettati')
        ax3.set_title("Distribuzione scattering\ncon sorgente all'orizzonte", color = 'slategrey')
        ax1.set_ylabel(r'Numero di fotoni per unità di superficie e tempo  $(\frac{fotoni}{s\cdot m^2})$')
        for ax in fig.get_axes():
            ax.set_xlabel("Lunghezza d'onda (nm)")
            ax.legend()

        plt.show()


        #####################################################################################################
        #Confronto fotoni diffusi per scattering di Rayleigh con Sirius allo zenith e all'orizzonte

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (8,9), gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle("Confronto distribuzione dei fotoni emessi e di quelli osservati\n considerando lo scattering di Rayleigh", color = 'slategrey')

        ax1.hist(l_emesse, bins = 100, color = 'navajowhite', label = 'Senza assorbimento')
        ax1.hist(l_scatter, bins = 100, color = 'seagreen', label = 'Scattering di Rayleigh con Sirius allo zenith', alpha = 0.5)
        ax1.hist(l_scatter_o, bins = 100, color = 'tomato', label = "Scattering di Rayleigh con Sirius all'orizzonte")

        #Plot della differenza tra i 10000 fotoni generati casualmente e quelli selezionati dall'hit or miss, proporzionale alla densità di fotoni diffusi
              
        y2 = n-n3
        y1 = n-n2
        X2, Y2 = np.meshgrid(bincenters3, y2)

        ax2.plot(bincenters3, n-n2, color = 'b', label = 'Scattering con Sirius allo zenith')
        ax2.plot(bincenters3, n-n3, color = 'black', label = "Scattering con Sirius all'orizzonte")
        ax2.imshow(X2, clim=clim, extent=(np.min(bincenters3), np.max(bincenters3), np.min(y1), np.max(y2)+10), cmap=spectralmap, aspect='auto')
        ax2.fill_between(bincenters3, y2, np.full(len(bincenters3), np.max(y2)+10), color='w')
        ax2.set_title('Andamento proporzionale alla densità di fotoni diffusi', fontsize = 10)
        ax2.set_xlabel("Lunghezza d'onda (nm)")
        ax1.set_ylabel(r'Densità di fotoni osservati $(\frac{fotoni}{s \cdot m^2})$')
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        for ax in fig.get_axes():
            ax.legend(loc = 'lower left')
        plt.show()

        ################################################################################################
        #Studio del flusso integrato in funzione dell'angolo di Sirius rispetto allo zenith

        theta = np.radians(np.linspace(0, 90, 50))
        valore=500*10**(-9)                                                                          
        simpson_int = np.empty(50)
        for i in range(0, 50):
            X_max = scipy.optimize.minimize(md.N_obs_opp, x0=valore, args=(t, theta[i]))        
            norm_ = md.N_obs(X_max.x[0], t, theta[i])
            y_base = md.N_obs_norm(l_emesse*10**(-9), t, theta[i], norm_)
            value = random.uniform(low = 0, high = np.max(y_base), size = len(l_emesse))
            mask = value <= y_base
            l_int = l_emesse[mask]
            y = y_base[mask]
            ind = np.argsort(l_int)
            l_int = l_int[ind]
            y = y[ind]
            hist = np.histogram(l_int, bins = 100)
            bincenters = (hist[1][:-1] + hist[1][1:])/2
            simpson_int[i] = integrate.simpson(hist[0], bincenters)

        plt.plot(np.degrees(theta), simpson_int, 'o', color = 'indianred')
        plt.xlabel('Angolo di Sirius rispetto allo zenith (°)')
        plt.ylabel('Flusso integrato di fotoni')
        plt.title("Flusso integrato dei fotoni in funzione\nall'angolo di Sirius rispetto allo Zenith", color = 'slategrey')
        plt.show()
      
    
    if args.rigel == True:
        print('TRAMONTO DI RIGEL')

        ########################################################################
        #Generazione fotoni con il metodo hit or miss

        #Plot dei fotoni generati vs quelli emessi con il metodo hit or miss
        plt.hist(l, bins = 100,  range=(380, 750), label='generati', color = 'rosybrown')
        n, bins1, p = plt.hist(stelle.get('Rigel')[0], bins = 100, range=(380, 750), label='selezionate', color = 'teal', alpha = 0.5)
        bincenters = (bins1[:-1]+bins1[1:])/2
        plt.xlabel("Lunghezza d'onda (m)")
        plt.ylabel(r'Densità di fotoni osservati $(\frac{fotoni}{s \cdot m^2})$')
        plt.legend()
        plt.title('Metodo hit or miss per la selezione di samples\nsecondo lo spettro del corpo nero', color = 'slategrey')
        plt.show()

        ####################################################################
        #Spettro di emissione

        #Plot dello spettro di emissione di Rigel usando la funzione corpo_nero riscalata ad 1
        f_emessi = stelle.get('Rigel')[1]
        l_emesse = stelle.get('Rigel')[0]
        t = stelle.get('Rigel')[2]
        y = f_emessi
        X, Y =  np.meshgrid(l_emesse, y)
        extent = (np.min(l_emesse), np.max(l_emesse), np.min(y), np.max(y)+0.1)
        
        
        #calcolo della lunghezza d'onda per cui si ha la massima emissione
        i = np.where(f_emessi == np.max(f_emessi))[0]
        l_max = l_emesse[i[0]]

        fig, axs = plt.subplots(1, 1, figsize=(6,5), tight_layout=True)
        plt.plot(l_emesse, f_emessi, '.', color='darkred')
        plt.axvline(x = l_max, label = 'λ_max = {:.5} nm'.format(l_max), color = 'slategrey')
        plt.imshow(X, clim=clim, extent=extent, cmap=spectralmap, aspect='auto')
        plt.xlabel("Lunghezza d'onda (nm)")
        plt.ylabel(r"Densità di fotoni emessi \nper lunghezza d'onda $(\frac{fotoni}{m^3})$")
        plt.title('Spettro di emissione di Rigel (Riscalato ad 1)', color = 'slategrey')
        plt.fill_between(l_emesse, f_emessi, np.full(len(l_emesse), np.max(y)+0.1), color='w')
        plt.legend()
        plt.show()

         ###############################################################################################7
        #Scattering di Rayleigh con sorgente allo zenith

        #Distribuzione dei fotoni considerando lo scattering di Rayleigh con Rigel allo Zenith
        theta = math.radians(0)
        valore=500*10**(-9)                                                                          

        X_max = scipy.optimize.minimize(md.N_obs_opp, x0=valore, args=(t, theta))        
        norm_z = md.N_obs(X_max.x[0], t, theta)     

        l0 = random.uniform(low = 380, high = 750, size = n_sample)
        N_s_z_base = md.N_obs_norm(l0*10**(-9), t, theta, norm_z)
        value = random.uniform(low = 0, high = np.max(N_s_z_base), size = n_sample)
        mask = value <= N_s_z_base
        l_scatter = l0[mask]
        l_scatter = np.sort(l_scatter)

        plt.title('Metodo hit or miss per la selezione di samples\n considerando lo scattering di Rayleigh con Rigel allo Zenith', color = 'slategrey')
        n0, b, p = plt.hist(l0, bins = 100, color = 'navajowhite', label = 'Generati')
        n2, bins2, p = plt.hist(l_scatter, bins = 100, color = 'tomato', label = 'Selezionati', alpha = 0.5)
        bincenters2 = (bins2[:-1] + bins2[1:])/2
        plt.xlabel("Lunghezza d'onda (nm)")
        plt.ylabel(r'Numero di fotoni per unità di superficie e tempo  $(\frac{fotoni}{s\cdot m^2})$')
        plt.yscale('log')
        plt.legend()
        plt.show()

        ###############################################################################################7
        #Scattering di Rayleigh con sorgente all'orizzonte

        #Distribuzione dei fotoni considerando lo scattering di Rayleigh con Rigel all'orizzonte
        theta = math.radians(90)
        valore=500*10**(-9)                                                                          

        X_max = scipy.optimize.minimize(md.N_obs_opp, x0=valore, args=(t, theta))        
        norm_o = md.N_obs(X_max.x[0], t, theta)     

        l2 = random.uniform(low = 380, high = 750, size = n_sample)
        N_s_o_base = md.N_obs_norm(l2*10**(-9), t, theta, norm_o)
        value = random.uniform(low = 0, high = np.max(N_s_o_base), size = n_sample)
        mask = value <= N_s_o_base
        l_scatter_o = l2[mask]
        l_scatter_o = np.sort(l_scatter_o)


        plt.title("Metodo hit or miss per la selezione di samples\n considerando lo scattering di Rayleigh con Rigel all'orizzonte", color = 'slategrey')
        n1, bins, p = plt.hist(l2, bins = 100, color = 'navajowhite', label = 'Generati')
        n3, bins3, p = plt.hist(l_scatter_o, bins = 100, color = 'tomato', label = "Selezionati", alpha = 0.5)
        bincenters3 = (bins3[:-1] + bins3[1:])/2
        plt.xlabel("Lunghezza d'onda (nm)")
        plt.ylabel(r'Numero di fotoni per unità di superficie e tempo  $(\frac{fotoni}{s\cdot m^2})$')
        plt.yscale('log')
        plt.legend()
        plt.show()

        #########################################################################################################
        #Confronto distribuzioni teoriche ed effettive

        #Calcolo gli integrali per normalizzare la distribuzione ed avere significato di probabilità
        x = np.linspace(380*10**(-9),750*10**(-9), 10000)
        y_1 = md.corpo_nero(x, t)
        norm1 = integrate.simpson(y_1,x)
        y_2 = md.N_obs(x, t, 0)
        norm2 = integrate.simpson(y_2, x)
        y_3 = md.N_obs(x, t, np.pi/2)
        norm3 = integrate.simpson(y_3, x)

        #Calcolo il numero di fotoni aspettati moltiplicando la probabilità di ogni bin per il numero di fotoni totali
        #Probabilità associata ai bin: (larghezza del bin)*(valore della distribuzione nel bincenter)
        base1 = (bincenters[1]-bincenters[0])*10**(-9)
        base2 = (bincenters2[1]-bincenters2[0])*10**(-9)
        base3 = (bincenters3[1]-bincenters3[0])*10**(-9)
        area_1 = base1*md.corpo_nero_norm(bincenters*10**(-9), t, norm1)                            
        area_2 = base2*md.N_obs_norm(bincenters2*10**(-9), t, 0, norm2)
        area_3 = base3*md.N_obs_norm(bincenters3*10**(-9), t, math.radians(90), norm3)
        dist_1 = area_1*len(l_emesse)
        dist_2 = area_2*len(l_scatter)
        dist_3 = area_3*len(l_scatter_o)

        #Confronto grafico fotoni osservati-fotoni aspettati
        fig, ((ax1), (ax2), (ax3)) = plt.subplots(1, 3, figsize = (9,8), sharey = True)

        fig.suptitle('Confronto distribuzioni teoriche ed effettive', color = 'slategrey')
        ax1.hist(l_emesse, bins = 100, color = 'seagreen', label = 'Osservati')
        ax1.plot(bincenters, dist_1, color = 'darkgreen', label = 'Aspettati')
        ax1.set_title('Distribuzione del corpo nero', color = 'slategrey')
        ax2.hist(l_scatter, bins = 100, color = 'skyblue', label = 'Osservati')
        ax2.plot(bincenters2, dist_2, color = 'navy', label = 'Aspettati')
        ax2.set_title('Distribuzione scattering\ncon sorgente allo Zenith', color = 'slategrey')
        ax3.hist(l_scatter_o, bins = 100, color = 'indianred', label = 'Osservati')
        ax3.plot(bincenters3, dist_3, color = 'maroon', label = 'Aspettati')
        ax3.set_title("Distribuzione scattering\ncon sorgente all'orizzonte", color = 'slategrey')
        ax1.set_ylabel(r'Numero di fotoni per unità di superficie e tempo  $(\frac{fotoni}{s\cdot m^2})$')
        for ax in fig.get_axes():
            ax.set_xlabel("Lunghezza d'onda (nm)")
            ax.legend()

        plt.show()


        #####################################################################################################
        #Confronto fotoni diffusi per scattering di Rayleigh con Rigel allo zenith e all'orizzonte

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (8,9), gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle("Confronto distribuzione dei fotoni emessi e di quelli osservati\n considerando lo scattering di Rayleigh", color = 'slategrey')

        ax1.hist(l_emesse, bins = 100, color = 'navajowhite', label = 'Senza assorbimento')
        ax1.hist(l_scatter, bins = 100, color = 'seagreen', label = 'Scattering di Rayleigh con Rigel allo zenith', alpha = 0.5)
        ax1.hist(l_scatter_o, bins = 100, color = 'tomato', label = "Scattering di Rayleigh con Rigel all'orizzonte")

        #Plot della differenza tra i 10000 fotoni generati casualmente e quelli selezionati dall'hit or miss, proporzionale alla densità di fotoni diffusi
              
        y2 = n-n3
        y1 = n-n2
        X2, Y2 = np.meshgrid(bincenters3, y2)

        ax2.plot(bincenters3, n-n2, color = 'b', label = 'Scattering con Rigel allo zenith')
        ax2.plot(bincenters3, n-n3, color = 'black', label = "Scattering con Rigel all'orizzonte")
        ax2.imshow(X2, clim=clim, extent=(np.min(bincenters3), np.max(bincenters3), np.min(y1), np.max(y2)+10), cmap=spectralmap, aspect='auto')
        ax2.fill_between(bincenters3, y2, np.full(len(bincenters3), np.max(y2)+10), color='w')
        ax2.set_title('Andamento proporzionale alla densità di fotoni diffusi', fontsize = 10)
        ax2.set_xlabel("Lunghezza d'onda (nm)")
        ax1.set_ylabel(r'Densità di fotoni osservati $(\frac{fotoni}{s \cdot m^2})$')
        ax1.set_yscale('log')
        ax2.set_yscale('log')
        for ax in fig.get_axes():
            ax.legend(loc = 'lower left')
        plt.show()

        ################################################################################################
        #Studio del flusso integrato in funzione dell'angolo di Rigel rispetto allo zenith

        theta = np.radians(np.linspace(0, 90, 50))
        valore=500*10**(-9)                                                                          
        simpson_int = np.empty(50)
        for i in range(0, 50):
            X_max = scipy.optimize.minimize(md.N_obs_opp, x0=valore, args=(t, theta[i]))        
            norm_ = md.N_obs(X_max.x[0], t, theta[i])
            y_base = md.N_obs_norm(l_emesse*10**(-9), t, theta[i], norm_)
            value = random.uniform(low = 0, high = np.max(y_base), size = len(l_emesse))
            mask = value <= y_base
            l_int = l_emesse[mask]
            y = y_base[mask]
            ind = np.argsort(l_int)
            l_int = l_int[ind]
            y = y[ind]
            hist = np.histogram(l_int, bins = 100)
            bincenters = (hist[1][:-1] + hist[1][1:])/2
            simpson_int[i] = integrate.simpson(hist[0], bincenters)

        plt.plot(np.degrees(theta), simpson_int, 'o', color = 'indianred')
        plt.xlabel('Angolo di Rigel rispetto allo zenith (°)')
        plt.ylabel('Flusso integrato di fotoni')
        plt.title("Flusso integrato dei fotoni in funzione\nall'angolo di Rigel rispetto allo Zenith", color = 'slategrey')
        plt.show()
      

if __name__ == "__main__":

    main()

