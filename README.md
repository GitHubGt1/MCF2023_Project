# MCF2023_Project

Progetto finale del corso di Metodi Computazionali per la Fisica 
Studentessa: Tosti Greta

**TRAMONTI**

Nella repository sono presenti 5 file oltre al README.md:

 - Tramonti.pdf: Spiegazione del progetto
 - observed_starX.csv: File di dati da analizzare
 - Modulo.py: Modulo python che implementa le funzioni necessarie ai programmi
 - Tramonti.py: Programma che implementa la simulazione di tramonti a partire dallo spettro del corpo nero per poi considerare lo scattering di Rayleigh al variare dell'angolo della sorgente di fotoni rispetto allo zenith.
 - StellaX.py: Programma che analizza i dati importati da observed_starX.csv eseguendo un fit dello spettro della stella considerando lo scattering di Rayleigh al fine di trovare la temperatura della stella.

Come eseguire i programmi:

1) _Tramonti.py_:
   Per prima cosa è necessario cambiare il pathname del file Modulo.py nell'import, mettendo il 	perorso effettivo dopo aver scaricato il file.
   
   Per eseguire il programma per prima cosa si deve scegliere la stella sorgente dei 		fotoni, le opzioni possibili sono:
   
   - Sole: da terminale usare il comando      _python3_ _Tramonti.py_ _--main_
   - Antares: da terminale usare il comando      _python3_ _Tramonti.py_ _--antares_
   - Sirius: da terminale usare il comando      _python3_ _Tramonti.py_ _--sirius_
   - Rigel: da terminale usare il comando      _python3_ _Tramonti.py_ _--rigel_
   
   Per ulteriori informazioni è sufficiente eseguire il comando:   _python3_ _Tramonti.py_ _--help_
   
2) _StellaX.py_:

   Come per il file precendente, cambiare il pathname nell'import del file Modulo.py.
   Per eseguire questo programma è anche necessario cambiare il pathname del file di dati che viene importato dopo averlo scaricato, inserendo quello effettivo.
   Successivamente è sufficiente runnare il programma:    _python3_ _StellaX.py_

