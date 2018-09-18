import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import sys

#############################
#Smooth Spectra
############################

#load the particle spectra files
spectra_pi = pd.read_csv(sys.argv[1] + "/dN_dpTdphidy_211.dat", sep='\t')
dN_dpTdphidy_pi = spectra_pi['dN_dpTdphidy']

spectra_k = pd.read_csv(sys.argv[1] + "/dN_dpTdphidy_321.dat", sep='\t')
dN_dpTdphidy_k = spectra_k['dN_dpTdphidy']

spectra_p = pd.read_csv(sys.argv[1] + "/dN_dpTdphidy_2212.dat", sep='\t')
dN_dpTdphidy_p = spectra_p['dN_dpTdphidy']

y = spectra_pi['y']
phip = spectra_pi['phip']
pT = spectra_pi['pT']

#spectra at midrapidity
dN_dpTdphidy_pi_mid = []
dN_dpTdphidy_k_mid = []
dN_dpTdphidy_p_mid = []
pT_mid = []

for i in range(0, len(dN_dpTdphidy_pi) ):
    if ( y[i] == 0 and phip[i] < 1.0e-2 ):
         dN_dpTdphidy_pi_mid.append( dN_dpTdphidy_pi[i] )
         dN_dpTdphidy_k_mid.append( dN_dpTdphidy_k[i] )
         dN_dpTdphidy_p_mid.append( dN_dpTdphidy_p[i] )
         pT_mid.append( pT[i] )

#find the normalization of dN/dpTdphidy and normalize it
pT_weights_file = pd.read_csv("tables/pT_gauss_table_left_align.dat")
pT_values = pT_weights_file['pT']
pT_weights = pT_weights_file['weight']
print("pT weights are ")
print(pT_weights)

phi_weights_file = pd.read_csv("tables/phi_gauss_table_left_align.dat")
phi_values = phi_weights_file['phi']
phi_weights = phi_weights_file['weight']
print("phi weights are ")
print(phi_weights)

y_weights_file = pd.read_csv("tables/y_riemann_table_11pt_left_align.dat")
y_values = y_weights_file['y']
y_weights = y_weights_file['weight']
print("y weights are ")
print(y_weights)

pi_norm = 0.0
k_norm = 0.0
p_norm = 0.0
for j in range(0, len(pT_weights)):
    pi_norm = pi_norm + pT_weights[j] * dN_dpTdphidy_pi_mid[j]
    k_norm = k_norm + pT_weights[j] * dN_dpTdphidy_k_mid[j]
    p_norm = p_norm + pT_weights[j] * dN_dpTdphidy_p_mid[j]

#rescale smooth spectra by normalization factor
for j in range(0, len(dN_dpTdphidy_pi_mid)):
    dN_dpTdphidy_pi_mid[j] = dN_dpTdphidy_pi_mid[j] / pi_norm
    dN_dpTdphidy_k_mid[j] = dN_dpTdphidy_k_mid[j] / k_norm
    dN_dpTdphidy_p_mid[j] = dN_dpTdphidy_p_mid[j] / p_norm

#find total yield from smooth ASSUMING OPTICAL GLAUBER
yield_pi_smooth = 0.0
yield_k_smooth = 0.0
yield_p_smooth = 0.0

for phi_w in range(0, len(phi_weights)):
    for y_w in range(0, len(y_weights)):
        yield_pi_smooth = yield_pi_smooth + pi_norm * phi_weights[phi_w] * y_weights[y_w]
        yield_k_smooth = yield_k_smooth + k_norm * phi_weights[phi_w] * y_weights[y_w]
        yield_p_smooth = yield_p_smooth + p_norm * phi_weights[phi_w] * y_weights[y_w]

##############################
#Sampled Particles
##############################
nevents = int(sys.argv[3]) - 1

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

#midrapidity
pi_pT_mid = []
k_pT_mid = []
p_pT_mid = []


#loop over particle lists
for idx in range(1, nevents + 1):
    particle_list = pd.read_csv(sys.argv[2] + "/results_" + str(idx) + "/particle_list.dat", sep=',')
    print("Reading file : " + sys.argv[2] + "/results_" + str(idx) + "/particle_list.dat")
    mcid = particle_list['mcid']
    E  = particle_list['E']
    px = particle_list['px']
    py = particle_list['py']
    pz = particle_list['pz']
    #3d momentum space lists
    for i in range(0, len(E) ):
        if ( mcid[i] == 211 ):
            pi_pT.append( math.sqrt( px[i]*px[i] + py[i]*py[i] ) )
            pi_y.append( 0.5 * math.log( (E[i] + pz[i]) / (E[i] - pz[i]) ) )
            pi_phi.append( math.atan( py[i] / px[i] ) )
        elif ( mcid[i] == 321 ):
            k_pT.append( math.sqrt( px[i]*px[i] + py[i]*py[i] ) )
            k_y.append( 0.5 * math.log( (E[i] + pz[i]) / (E[i] - pz[i]) ) )
            k_phi.append( math.atan( py[i] / px[i] ) )
        elif ( mcid[i] == 2212 ):
            p_pT.append( math.sqrt( px[i]*px[i] + py[i]*py[i] ) )
            p_y.append( 0.5 * math.log( (E[i] + pz[i]) / (E[i] - pz[i]) ) )
            p_phi.append( math.atan( py[i] / px[i] ) )
#end of loop over particle lists

#the range of rapidity to integrate over
ymax = 0.5
for i in range(0, len(pi_pT) ):
    if ( abs(pi_y[i]) < ymax ):
        pi_pT_mid.append( pi_pT[i] )
for i in range(0, len(k_pT) ):
    if ( abs(k_y[i]) < ymax ):
        k_pT_mid.append( k_pT[i] )
for i in range(0, len(p_pT) ):
    if ( abs(p_y[i]) < ymax ):
        p_pT_mid.append( p_pT[i] )

#pT bins
delta_pT = 0.2
pT_bins_eqwidth = []
for j in range(0, 20):
    pT_bins_eqwidth.append( j * delta_pT)

#normalization factor
norm = nevents
print("nevents is " + str(nevents))
#print("norm factor is " + str(norm))


#plot pion
nPi, binsPi, patchesPi = plt.hist(pi_pT_mid, bins = pT_bins_eqwidth, normed=True)
plt.plot(pT_mid, dN_dpTdphidy_pi_mid)
plt.title("Pion spectra midrapidity")
plt.xlabel("pT (GeV)")
plt.show()

#plot kaon
nK, binsK, patchesK = plt.hist(k_pT_mid, bins = pT_bins_eqwidth, normed=True)
plt.plot(pT_mid, dN_dpTdphidy_k_mid)
plt.title("Kaon spectra midrapidity")
plt.xlabel("pT (GeV)")
plt.show()

#plot proton
nP, binsP, patchesP = plt.hist(p_pT_mid, bins = pT_bins_eqwidth, normed=True)
plt.plot(pT_mid, dN_dpTdphidy_p_mid)
plt.title("Proton spectra midrapidity")
plt.xlabel("pT (GeV)")
plt.show()

#compare total yields
yield_pi_sample = len(pi_pT) / nevents
yield_k_sample = len(k_pT) / nevents
yield_p_sample = len(p_pT) / nevents 

ratio_pi_yield = yield_pi_sample / yield_pi_smooth
ratio_k_yield = yield_k_sample / yield_k_smooth
ratio_p_yield = yield_p_sample / yield_p_smooth

print("Ratios of yields : pion " + str(ratio_pi_yield) + ", kaon " + str(ratio_k_yield) + ", proton " + str(ratio_p_yield))

#examine dN/dphi
plt.hist(pi_phi, bins = 'auto', normed=True)
plt.title("Pion 1/N_part dN/dphi")
plt.xlabel("phi")
plt.show()

plt.hist(k_phi, bins = 'auto', normed=True)
plt.title("Kaon 1/N_part dN/dphi")
plt.xlabel("phi")
plt.show()

plt.hist(p_phi, bins = 'auto', normed=True)
plt.title("Proton 1/N_part dN/dphi")
plt.xlabel("phi")
plt.show()

#examine dN/dy
plt.hist(pi_y, bins = 'auto', normed=True)
plt.title("Pion 1/N_part dN/y")
plt.xlabel("y")
plt.show()

plt.hist(k_y, bins = 'auto', normed=True)
plt.title("Kaon 1/N_part dN/dy")
plt.xlabel("y")
plt.show()

plt.hist(p_y, bins = 'auto', normed=True)
plt.title("Proton 1/N_part dN/dy")
plt.xlabel("y")
plt.show()
