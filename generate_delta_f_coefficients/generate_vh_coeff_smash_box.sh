####################################################

echo "Calculating df coefficients for Smash Box HRG"

cd smash_box/df_vh_dimensionless

echo "removing old smash df coefficients and pdg.dat"
#remove old coefficient tables and pdg file
rm -r smash_box
rm pdg.dat

echo "copying in new smash pdg.dat from /PDG"
#copy in new pdg file and rename it as pdg.dat
cp ../../../PDG/pdg_box.dat .
mv pdg_box.dat pdg.dat

#make a directory to store the new coefficients
mkdir smash_box

echo "compiling"
#clean and compile
sh cleanMake.sh

echo "running code to generate vh coeffs"
#run the executable to generate new tables
./deltaf_table

echo "copying new tables to /deltaf_coefficients"
cd ..
#copy the tables back to the deltaf_coeffs directory
cp -R df_vh_dimensionless/smash_box ../../deltaf_coefficients/vh/.

# return to original directory
cd ..
