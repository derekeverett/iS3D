
rm "average_thermodynamic_quantities.dat"
rm -r results
mkdir results

rm iS3D.e
cd src
rm iS3D.e
cd cpp
rm iS3D.e
rm -r obj
mkdir obj
cd ../../

make iS3D.e

