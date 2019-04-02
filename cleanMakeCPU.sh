
rm "average_thermodynamic_quantities.dat"
rm -r results
mkdir results
mkdir results/vn
mkdir results/vn_continuous
mkdir results/spacetime_distribution
mkdir results/momentum_distribution
mkdir results/dN_dy

rm iS3D.e
cd src
rm iS3D.e
cd cpp
rm iS3D.e
rm -r obj
mkdir obj
cd ../../

make iS3D.e

