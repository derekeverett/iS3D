import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import sys

sample_dir_1 = sys.argv[1]
sample_dir_2 = sys.argv[2]
nsamples = int(sys.argv[3])

#the range of rapidity cut
ymax = 1.0

#species dependent info
#pion 211
pi_pT_1  = []
pi_y_1 = []

#kaon 321
k_pT_1  = []
k_y_1 = []

#proton 2212
p_pT_1  = []
p_y_1 = []

#all particles
all_t_1 = []
all_x_1 = []

#after cut on pseudorap or rap
pi_pT_mid_1 = []
k_pT_mid_1 = []
p_pT_mid_1 = []

all_t_mid_1 = []
all_x_mid_1 = []

#species dependent info
#pion 211
pi_pT_2  = []
pi_y_2 = []

#kaon 321
k_pT_2  = []
k_y_2 = []

#proton 2212
p_pT_2  = []
p_y_2 = []

#all particles
all_t_2 = []
all_x_2 = []

#after cut on pseudorap or rap
pi_pT_mid_2 = []
k_pT_mid_2 = []
p_pT_mid_2 = []

all_t_mid_2 = []
all_x_mid_2 = []

#GET PARTICLE LIST FROM FIRST MODULE
for sample in range(0, nsamples):
    hadron_file = sample_dir_1 + 'sample_' + str(sample) + '/results/particle_list_osc_1.dat'
    print("Reading file : " + hadron_file )

    df = pd.read_csv( hadron_file, delimiter=' ')
    df_mcid = df['pid']
    df_px = df['px']
    df_py = df['py']
    df_pz = df['pz']
    df_E = df['E']
    df_m = df['m']
    df_x = df['x']
    df_z = df['z']
    df_t = df['t']

    mcid_1 = df_mcid.tolist()
    px_1 = df_px.tolist()
    py_1 = df_py.tolist()
    pz_1 = df_pz.tolist()
    E_1 = df_E.tolist()
    m_1 = df_m.tolist()
    x_1 = df_x.tolist()
    z_1 = df_z.tolist()
    t_1 = df_t.tolist()

    #3d momentum space lists
    for i in range(0, len(E_1) ):
        if ( mcid_1[i] == 211 ):
            pi_pT_1.append( math.sqrt( px_1[i]*px_1[i] + py_1[i]*py_1[i] ) )
            energy = E_1[i]
            pz =  pz_1[i]
            arg = (energy + pz) / (energy - pz)
            if (pz == 0):
                y = 0
            else:
                y = 0.5 * math.log( abs(arg) )
            pi_y_1.append( y )

        elif ( mcid_1[i] == 321 ):
            k_pT_1.append( math.sqrt( px_1[i]*px_1[i] + py_1[i]*py_1[i] ) )
            energy = E_1[i]
            pz =  pz_1[i]
            arg = (energy + pz) / (energy - pz)
            if (pz == 0):
                y = 0
            else:
                y = 0.5 * math.log( abs(arg) )
            k_y_1.append( y )

        elif ( mcid_1[i] == 2212 ):
            p_pT_1.append( math.sqrt( px_1[i]*px_1[i] + py_1[i]*py_1[i] ) )
            energy = E_1[i]
            pz =  pz_1[i]
            arg = (energy + pz) / (energy - pz)
            if (pz == 0):
                y = 0
            else:
                y = 0.5 * math.log( abs(arg) )
            p_y_1.append( y )

    #spacetime info with pseudorap cut
    for i in range(0, len(t_1)):
        pz = pz_1[i]
        energy = E_1[i]
        arg = (energy + pz) / (energy - pz)
        if (pz == 0):
            y = 0
        else:
            y = 0.5 * math.log( abs(arg) )
        if ( abs( y ) < ymax ):
            all_t_mid_1.append( t_1[i] )
            all_x_mid_1.append( x_1[i] )

#pseudorap cuts on momentum space info
for i in range(0, len(pi_pT_1) ):
    if ( abs(pi_y_1[i]) < ymax ):
        pi_pT_mid_1.append( pi_pT_1[i] )
for i in range(0, len(k_pT_1) ):
    if ( abs(k_y_1[i]) < ymax ):
        k_pT_mid_1.append( k_pT_1[i] )
for i in range(0, len(p_pT_1) ):
    if ( abs(p_y_1[i]) < ymax ):
        p_pT_mid_1.append( p_pT_1[i] )


#GET PARTICLE LIST FROM SECOND MODULE
hadron_file = sample_dir_2
print("Reading file : " + hadron_file )

df = pd.read_csv( hadron_file, delimiter='\t')
df_mcid = df['pid']
df_px = df['px']
df_py = df['py']
df_pz = df['pz']
df_E = df['E']
df_m = df['m']
df_x = df['x']
df_z = df['z']
df_t = df['t']

mcid_2 = df_mcid.tolist()
px_2 = df_px.tolist()
py_2 = df_py.tolist()
pz_2 = df_pz.tolist()
E_2 = df_E.tolist()
m_2 = df_m.tolist()
x_2 = df_x.tolist()
z_2 = df_z.tolist()
t_2 = df_t.tolist()

#3d momentum space lists
for i in range(0, len(E_2) ):
    if ( mcid_2[i] == 211 ):
        pi_pT_2.append( math.sqrt( px_2[i]*px_2[i] + py_2[i]*py_2[i] ) )
        energy = E_2[i]
        pz =  pz_2[i]
        arg = (energy + pz) / (energy - pz)
        if (pz == 0):
            y = 0
        else:
            y = 0.5 * math.log( abs(arg) )
        pi_y_2.append( y )

    elif ( mcid_2[i] == 321 ):
        k_pT_2.append( math.sqrt( px_2[i]*px_2[i] + py_2[i]*py_2[i] ) )
        energy = E_2[i]
        pz =  pz_2[i]
        arg = (energy + pz) / (energy - pz)
        if (pz == 0):
            y = 0
        else:
            y = 0.5 * math.log( abs(arg) )
        k_y_2.append( y )

    elif ( mcid_2[i] == 2212 ):
        p_pT_2.append( math.sqrt( px_2[i]*px_2[i] + py_2[i]*py_2[i] ) )
        energy = E_2[i]
        pz =  pz_2[i]
        arg = (energy + pz) / (energy - pz)
        if (pz == 0):
            y = 0
        else:
            y = 0.5 * math.log( abs(arg) )
        p_y_2.append( y )

#spacetime info with pseudorap cut
for i in range(0, len(t_2)):
    pz = pz_2[i]
    energy = E_2[i]
    arg = (energy + pz) / (energy - pz)
    if (pz == 0):
        y = 0
    else:
        y = 0.5 * math.log( abs(arg) )
    if ( abs( y ) < ymax ):
        all_t_mid_2.append( t_2[i] )
        all_x_mid_2.append( x_2[i] )

#pseudorap cuts on momentum space info
for i in range(0, len(pi_pT_2) ):
    if ( abs(pi_y_2[i]) < ymax ):
        pi_pT_mid_2.append( pi_pT_2[i] )
for i in range(0, len(k_pT_2) ):
    if ( abs(k_y_2[i]) < ymax ):
        k_pT_mid_2.append( k_pT_2[i] )
for i in range(0, len(p_pT_2) ):
    if ( abs(p_y_2[i]) < ymax ):
        p_pT_mid_2.append( p_pT_2[i] )


#pT bins
delta_pT = 0.2
pT_bins_eqwidth = []
for j in range(0, 20):
    pT_bins_eqwidth.append( j * delta_pT)
np.savetxt("plots/pT_bins.dat", pT_bins_eqwidth)

#time bins
delta_t = 0.2
t_bins = []
for j in range(-10, 10):
    t_bins.append( j * delta_t)
np.savetxt("plots/t_bins.dat", t_bins)

#x bins
delta_x = 1.0
x_bins = []
for j in range(-10, 10):
    x_bins.append( j * delta_x)
np.savetxt("plots/x_bins.dat", x_bins)

#y bins
delta_y = 0.4
y_bins = []
for j in range(-15, 15):
    y_bins.append( j * delta_y)
np.savetxt("plots/y_bins.dat", y_bins)

#examine dN/dpT
nPi_1, binsPi_1, patchesPi_1 = plt.hist(pi_pT_mid_1, bins = pT_bins_eqwidth, normed=False)
plt.title("Pion dN/dpT")
plt.xlabel("pT (GeV)")
plt.savefig('plots/211_spectra_1.pdf')
#plt.show()
plt.close()
np.savetxt("plots/pi_dN_dpT_1.dat", nPi_1)

#examine dN/dphi
#n_pi_phi_1, bins_pi_phi_1, patches_pi_phi_1 = plt.hist(pi_phi_mid, bins = phi_bins, normed=False)
#plt.title("Pion dN/dphi")
#plt.xlabel("phi")
#plt.savefig('plots/211_dNdphi_1.pdf')
#plt.show()
#plt.close()
#np.savetxt("plots/pi_dN_dphi_1.dat", n_pi_phi)

#examine dN/dy
n_pi_y_1, bins_pi_y_1, patches_pi_y_1 = plt.hist(pi_y_1, bins = y_bins, normed=False)
plt.title("Pion dN/dy")
plt.xlabel("y")
plt.savefig('plots/211_dNdy_1.pdf')
#plt.show()
plt.close()
np.savetxt("plots/211_dN_dy_1.dat", n_pi_y_1)

#examine dN/dt
n_t_1, bins_t_1, patches_t_1 = plt.hist(all_t_mid_1, bins = t_bins, normed=False)
plt.title("dN / dt")
plt.xlabel("t")
plt.savefig('plots/dNdt_1.pdf')
#plt.show()
plt.close()
np.savetxt("plots/dN_dt_1.dat", n_t_1)

#examine dN/dx
n_x_1, bins_x_1, patches_x_1 = plt.hist(all_x_mid_1, bins = x_bins, normed=False)
plt.title("dN / dx")
plt.xlabel("x")
plt.savefig('plots/dN_dx_1.pdf')
#plt.show()
plt.close()
np.savetxt("plots/dN_dx_1.dat", n_x_1)


## SECOND MODULE 

#examine dN/dpT
nPi_2, binsPi_2, patchesPi_2 = plt.hist(pi_pT_mid_2, bins = pT_bins_eqwidth, normed=False)
plt.title("Pion dN/dpT")
plt.xlabel("pT (GeV)")
plt.savefig('plots/211_spectra_2.pdf')
#plt.show()
plt.close()
np.savetxt("plots/pi_dN_dpT_2.dat", nPi_2)

#examine dN/dphi
#n_pi_phi_1, bins_pi_phi_1, patches_pi_phi_1 = plt.hist(pi_phi_mid, bins = phi_bins, normed=False)
#plt.title("Pion dN/dphi")
#plt.xlabel("phi")
#plt.savefig('plots/211_dNdphi_1.pdf')
#plt.show()
#plt.close()
#np.savetxt("plots/pi_dN_dphi_1.dat", n_pi_phi)

#examine dN/dy
n_pi_y_2, bins_pi_y_2, patches_pi_y_2 = plt.hist(pi_y_2, bins = y_bins, normed=False)
plt.title("Pion dN/dy")
plt.xlabel("y")
plt.savefig('plots/211_dNdy_2.pdf')
#plt.show()
plt.close()
np.savetxt("plots/211_dN_dy_2.dat", n_pi_y_2)

#examine dN/dt
n_t_2, bins_t_2, patches_t_2 = plt.hist(all_t_mid_2, bins = t_bins, normed=False)
plt.title("dN / dt")
plt.xlabel("t")
plt.savefig('plots/dNdt_2.pdf')
#plt.show()
plt.close()
np.savetxt("plots/dN_dt_2.dat", n_t_2)

#examine dN/dx
n_x_2, bins_x_2, patches_x_2 = plt.hist(all_x_mid_2, bins = x_bins, normed=False)
plt.title("dN / dx")
plt.xlabel("x")
plt.savefig('plots/dN_dx_2.pdf')
#plt.show()
plt.close()
np.savetxt("plots/dN_dx_2.dat", n_x_2)
