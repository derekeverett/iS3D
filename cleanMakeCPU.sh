export OMP_NUM_THREADS="$1"

rm -r results
mkdir results

rm iS.e
cd src
rm iS.e
cd cpp
rm iS.e
rm -r obj
mkdir obj
cd ../../

make iS.e

