
rm "average_thermodynamic_quantities.dat"
rm -r results
mkdir results
mkdir results/continuous
mkdir results/sampled
mkdir results/sampled/vn
mkdir results/sampled/dN_taudtaudy
mkdir results/sampled/dN_2pirdrdy
mkdir results/sampled/dN_dphisdy
mkdir results/sampled/dN_2pipTdpTdy
mkdir results/sampled/dN_dphipdy
mkdir results/sampled/dN_dy
mkdir results/sampled/dN_deta

rm iS3D.e
cd src
rm iS3D.e
cd cpp
rm iS3D.e
rm -r obj
mkdir obj
cd ../../

make iS3D.e

