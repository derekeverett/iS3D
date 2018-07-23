import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

#load the particle spectra file
particle_list = pd.read_csv('results/dN_dpTdphidy.dat', sep='\t')
y = particle_list['y']
phip = particle_list['phip']
pT = particle_list['pT']

dN_dpTdphidy = particle_list['dN_dpTdphidy']

#spectra at midrapidity
dN_dpTdphidy_mid = []
pT_mid = []

for i in range(1, len(dN_dpTdphidy) ):
    if ( y[i] == 0 and phip[i] < 1.0e-2 ):
         dN_dpTdphidy_mid.append( dN_dpTdphidy[i] )
         pT_mid.append( pT[i] )

plt.title("Pion spectra midrapidity")
plt.xlabel("pT (GeV)")
plt.plot(pT_mid, dN_dpTdphidy_mid)
plt.show()
