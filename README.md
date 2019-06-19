iS3D (c) Mike McNelis, Derek Everett, Sameed Pervaiz and Lipei Du.

## Purpose
This code can read in a freeze out surface from 3+1D viscous hydro or anisotropic
viscous hydro and calculate 3D smooth particle spectra or a sampled particle list.
The structure is based on iSpectra, the Cooper Frye code in the iEBE heavy ion
event generator (Chun Shen, Zhi Qiu).  

## Installation

To compile iS3D, one can do 

    mkdir build && cd build
    cmake ..
    make
    make install

## Usage 

To run iS3D

    ./iS3D

or

    sh runCPU.sh <num_threads>

where <num_threads> is the number of cpu threads.

The freezeout surface is read from input/surface.dat, or from memory depending on how the wrapper is called.
By default, input/surface.dat contains a toy freezeout surface with one cell. 
See parameters.dat for a list of compatible formats.

The results will be written in the `results` directory, so this directory must exist at runtime.
