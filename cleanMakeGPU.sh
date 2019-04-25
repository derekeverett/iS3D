
rm iS3D_GPU.e
cd src
rm iS3D_GPU.e
cd cuda
rm iS3D_GPU.e
rm -r obj
mkdir obj
cd ../../

rm -r results
mkdir results
mkdir results/continuous

make iS3D_GPU.e

