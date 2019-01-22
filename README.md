iS3D (c) Derek Everett, Mike McNelis, Sameed Pervaiz and Lipei Du
This code can read in a freeze out surface from 3+1D viscous hydro or anisotropic
viscous hydro and calculate 3D smooth particle spectra or a sampled particle list.
The structure is based on iSpectra, the Cooper Frye code in the iEBE heavy ion
event generator (Chun Shen, Zhi Qiu).  


Building

to build iS3D, one can do:

$mkdir build && cd build
$cmake ..
$make
$make install

Running

To run iS3D, do
$./iS3D

This assumes that there is a freezeout surface input/surface.dat that exists and has the correct format
See parameters.dat for a list of compatible formats

The results will be written in the /results directory, so this directory must exist
