export OMP_NUM_THREADS="$1"
rm iS.e
rm src/iS.e
cd src
make clean
cd ..
make
