export OMP_NUM_THREADS="$1"
rm iS_GPU.e
cd src
rm iS.e
cd cuda
rm iS.e
rm -r obj
mkdir obj
cd ../../

rm -r results
mkdir results

make iS_GPU.e

