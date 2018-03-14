export OMP_NUM_THREADS="$1"
#rm iS.e
#rm iS_GPU.e
#cd src
#rm iS.e
#cd cuda
#rm iS.e
#rm -r obj
#mkdir obj
#cd ../../
rm iS.e
cd src
rm iS.e
cd cpp
rm iS.e
rm -r obj
mkdir obj
cd ../../
make iS.e
#make iS_GPU.e

