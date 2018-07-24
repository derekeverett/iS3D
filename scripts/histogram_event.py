import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import sys

#load the particle list
particle_list = pd.read_csv(sys.argv[1], sep=',')

mcid = particle_list['mcid']

#spacetime info 
tau = particle_list['tau']
x   = particle_list['x']
eta = particle_list['eta']
#momentum info 
E  = particle_list['E']
px = particle_list['px']
py = particle_list['py']
pz = particle_list['pz']

#species dependent info 
#pion 211
pi_pT  = []
pi_y   = []
pi_phi = [] 

#kaon 321
k_pT  = []
k_y   = []
k_phi = [] 

#proton 2212
p_pT  = []
p_y   = []
p_phi = [] 

#3d momentum space lists
for i in range(1, len(E) ):
    if ( mcid[i] == 211 ):
        pi_pT.append( math.sqrt( px[i]*px[i] + py[i]*py[i] ) )
        pi_y.append( 0.5 * math.log( (E[i] + pz[i]) / (E[i] - pz[i]) ) )
        pi_phi.append( math.atan( py[i] / px[i] ) )
    if ( mcid[i] == 321 ):
        k_pT.append( math.sqrt( px[i]*px[i] + py[i]*py[i] ) )
        k_y.append( 0.5 * math.log( (E[i] + pz[i]) / (E[i] - pz[i]) ) )
        k_phi.append( math.atan( py[i] / px[i] ) )
    if ( mcid[i] == 2212 ):
        p_pT.append( math.sqrt( px[i]*px[i] + py[i]*py[i] ) )
        p_y.append( 0.5 * math.log( (E[i] + pz[i]) / (E[i] - pz[i]) ) )
        p_phi.append( math.atan( py[i] / px[i] ) )

#midrapidity
pi_pT_mid = []
k_pT_mid = []
p_pT_mid = []

for i in range(1, len(pi_pT) ):
    if ( abs(pi_y[i]) < 1.0 ):
        pi_pT_mid.append( pi_pT[i] )

for i in range(1, len(k_pT) ):
    if ( abs(k_y[i]) < 1.0 ):
        k_pT_mid.append( k_pT[i] )

for i in range(1, len(p_pT) ):
    if ( abs(p_y[i]) < 1.0 ):
        p_pT_mid.append( p_pT[i] )

#histogram of particle yields
plt.hist(mcid, bins='auto')
plt.title("Particle Yields")
plt.xlabel("MC ID")
plt.show()

#histogram of tau (proper time of production)
plt.hist(tau, bins='auto')
plt.title("Proper time of particle production")
plt.xlabel("tau (fm/c)")
plt.show()

#pion spectra at midrapidity
plt.hist(pi_pT_mid, bins='auto')
plt.title("Pion spectra midrapidity")
plt.xlabel("pT (GeV)")
plt.show()

#pion spectra at midrapidity
plt.hist(k_pT_mid, bins='auto')
plt.title("Kaon spectra midrapidity")
plt.xlabel("pT (GeV)")
plt.show()

#pion spectra at midrapidity
plt.hist(p_pT_mid, bins='auto')
plt.title("Proton spectra midrapidity")
plt.xlabel("pT (GeV)")
plt.show()

