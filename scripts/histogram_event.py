import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#load the particle list
#dtypes = ['int', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float']
particle_list = pd.read_csv('results/particle_list.dat', sep=',')
#mcid = pd.read_csv('results/particle_list.dat')['mcid']

#mcid = np.loadtxt('results/particle_list.dat', delimiter='\t', usecols=(0,) )


mcid = particle_list['mcid']

#histogram of particle yields
plt.hist(mcid, bins='auto')
plt.title("Particle Yields")
plt.show()
