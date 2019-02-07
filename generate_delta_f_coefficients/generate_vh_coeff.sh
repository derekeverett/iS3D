
cd df_vh_dimensionless

echo "removing old coeffs and pdg.dat"
#remove old coefficient tables and pdg file
rm -r vh
rm pdg.dat

echo "copying in new pdg.dat from /PDG"
#copy in new pdg file
cp ../../PDG/pdg.dat .

#make a dir to store the new coefficients
mkdir vh

echo "compiling"
#clean and compile
sh cleanMake.sh

echo "running code to generate vh coeffs"
#run the executable to generate new tables
./deltaf_table

echo "copying new tables to /deltaf_coefficients"
cd ..
#copy the tables back to the deltaf_coeffs dir
cp -R df_vh_dimensionless/vh ../deltaf_coefficients/. 
