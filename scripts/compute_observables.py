#taken from run-events script in hic-eventgen
#https://github.com/Duke-QCD/hic-eventgen

import argparse
from contextlib import contextmanager
import datetime
from itertools import chain, groupby, repeat
import logging
import math
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import numpy as np

# fully specify numeric data types, including endianness and size, to
# ensure consistency across all machines
float_t = '<f8'
int_t = '<i8'
complex_t = '<c16'

# species (name, ID) for identified particle observables
species = [
('pion', 211),
('kaon', 321),
('proton', 2212),
('Lambda', 3122),
('Sigma0', 3212),
('Xi', 3312),
('Omega', 3334),
]

# UrQMD raw particle format
parts_dtype = [
('sample', int),
('ID', int),
('charge', int),
('pT', float),
('ET', float),
('mT', float),
('phi', float),
('y', float),
('eta', float)
]

# results "array" (one element)
# to be overwritten for each event
results = np.empty((), dtype=[
('initial_entropy', float_t),
('nsamples', int_t),
('dNch_deta', float_t),
('dET_deta', float_t),
('dN_dy', [(s, float_t) for (s, _) in species]),
('mean_pT', [(s, float_t) for (s, _) in species]),
('pT_fluct', [('N', int_t), ('sum_pT', float_t), ('sum_pTsq', float_t)]),
('flow', [('N', int_t), ('Qn', complex_t, 8)]),
])

# read final particle data
with open('final_particles.dat', 'rb') as f:
    nsamples = 1

    # partition UrQMD file into oversamples
    groups = groupby(f, key=lambda l: l.startswith(b'#'))
    samples = filter(lambda g: not g[0], groups)

    # iterate over particles and oversamples
    parts_iter = (
    tuple( ( nsample, *l.split() ) )
    for nsample, (header, sample) in enumerate(samples, start=1)
    for l in sample
    )

    parts = np.fromiter(parts_iter, dtype=parts_dtype)

    print("computing observables")
    charged = (parts['charge'] != 0)
    abs_eta = np.fabs(parts['eta'])

    results['dNch_deta'] = \
    np.count_nonzero(charged & (abs_eta < .5)) / nsamples

    ET_eta = .6
    results['dET_deta'] = \
    parts['ET'][abs_eta < ET_eta].sum() / (2*ET_eta) / nsamples

    abs_ID = np.abs(parts['ID'])
    midrapidity = (np.fabs(parts['y']) < .5)

    pT = parts['pT']
    phi = parts['phi']

    for name, i in species:
        cut = (abs_ID == i) & midrapidity
        N = np.count_nonzero(cut)
        results['dN_dy'][name] = N / nsamples
        results['mean_pT'][name] = (0. if N == 0 else pT[cut].mean())
        pT_alice = pT[charged & (abs_eta < .8) & (.15 < pT) & (pT < 2.)]
        results['pT_fluct']['N'] = pT_alice.size
        results['pT_fluct']['sum_pT'] = pT_alice.sum()
        results['pT_fluct']['sum_pTsq'] = np.inner(pT_alice, pT_alice)

        phi_alice = phi[charged & (abs_eta < .8) & (.2 < pT) & (pT < 5.)]
        results['flow']['N'] = phi_alice.size
        results['flow']['Qn'] = [
        np.exp(1j*n*phi_alice).sum()
        for n in range(1, results.dtype['flow']['Qn'].shape[0] + 1)
        ]
