####################################################

echo "calculating viscous hydro df coefficients for UrQMD HRG"

####################################################

cd urqmd/df_vh_dimensionless

echo "removing old UrQMD vh df coefficients and pdg.dat"
#remove old coefficient tables and pdg file
rm -r urqmd
rm pdg.dat

echo "copying in new smash pdg.dat from /PDG"
#copy in new pdg file and rename it as pdg.dat
cp ../../../PDG/pdg-urqmd_v3.3+.dat .
mv pdg-urqmd_v3.3+.dat pdg.dat

#make a directory to store the new coefficients
mkdir urqmd

echo "compiling"
#clean and compile
sh cleanMake.sh

echo "running code to generate vh df coefficients"
#run the executable to generate new tables
./deltaf_table

echo "copying new tables to /deltaf_coefficients/vh/"
cd ..
#copy the tables back to the deltaf_coeffs directory
cp -R df_vh_dimensionless/urqmd ../../deltaf_coefficients/vh/.

# return to original directory
cd ..
