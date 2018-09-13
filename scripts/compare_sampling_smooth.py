import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import sys

#############################
#Smooth Spectra
############################

#load the particle spectra file
spectra = pd.read_csv(sys.argv[1], sep='\t')
y = spectra['y']
phip = spectra['phip']
pT = spectra['pT']
dN_dpTdphidy = spectra['dN_dpTdphidy']

#spectra at midrapidity
dN_dpTdphidy_mid = []
pT_mid = []

for i in range(0, len(dN_dpTdphidy) ):
    if ( y[i] == 0 and phip[i] < 1.0e-2 ):
         dN_dpTdphidy_mid.append( dN_dpTdphidy[i] )
         pT_mid.append( pT[i] )

#plt.title("Pion spectra midrapidity")
#plt.xlabel("pT (GeV)")
plt.plot(pT_mid, dN_dpTdphidy_mid)
#plt.show()


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
        if ( mcid[i] == 321 ):
            k_pT.append( math.sqrt( px[i]*px[i] + py[i]*py[i] ) )
            k_y.append( 0.5 * math.log( (E[i] + pz[i]) / (E[i] - pz[i]) ) )
            k_phi.append( math.atan( py[i] / px[i] ) )
        if ( mcid[i] == 2212 ):
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
pT_bins = [0,.0072, .038, .094, .175, .28, .42, .58, .78, 1.01, 1.3, 1.6, 1.97, 2.4, 2.96, 3.7]

#normalization factor
#norm = nevents * 2.0 * math.pi
norm = nevents
print("nevents is " + str(nevents))
print("norm factor is " + str(norm))

#pion spectra at midrapidity
#heights,bins,histo = plt.hist(pi_pT_mid, bins=pT_bins)
"""
#use numpy hist
hist, bins = np.histogram(pi_pT_mid, bins = pT_bins)
norm = nevents * 2.0 * math.pi
#norm = 1.0
hist = [ float(n)/norm for n in hist]
center = []
width = []
for j in range(1, len(pT_bins)):
    center.append( pT_bins[j] )
    width.append( (pT_bins[j] - pT_bins[j-1]) / 2.0 )

plt.bar(center, hist, align = 'center', width = width, color='y', alpha=0.6)
plt.title("Pion spectra midrapidity")
plt.xlabel("pT (GeV)")
plt.show()
"""

"""
#use numpy hist and bar chart after rescaling
hist, bins = np.histogram(p_pT_mid, bins = pT_bins)
print("norm factor is " + str(norm))
print("nevents is " + str(nevents))
print("hist before norm")
print(hist)
hist = [ float(n)/norm for n in hist]
print("hist after norm")
print(hist)
center = []
width = []
for j in range(1, len(pT_bins)):
    center.append( pT_bins[j] )
    width.append( (pT_bins[j] - pT_bins[j-1]) / 2.0 )

plt.bar(center, hist, align = 'center', width = width, color='y', alpha=0.6)
"""

#or use plt.hist
weights = []
for j in range(0, len(pi_pT_mid)):
    weights.append(1.0 / norm)
print( "histogram weighted by " + str(1.0 / norm) )

#n, bins, patches = plt.hist(p_pT_mid, bins = pT_bins)
n, bins, patches = plt.hist(pi_pT_mid, bins = pT_bins, weights = weights)

plt.title("Pion spectra midrapidity")
plt.xlabel("pT (GeV)")
plt.show()

"""
#check the ratio of sampling to smooth
ratio  = []
for j in range(0, len(hist)):
    ratio.append(hist[j] / dN_dpTdphidy_mid[j])
plt.plot(pT_mid, ratio)
plt.title("ratio of sampled / smooth")
plt.xlabel("pT (GeV)")
plt.show()
"""
