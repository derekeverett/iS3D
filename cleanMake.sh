export OMP_NUM_THREADS="$1"
rm iS.e
rm iS_GPU.e
cd src
rm iS.e
cd cuda
rm iS.e
rm -r obj
mkdir obj
cd ../..
make iS_GPU.e
