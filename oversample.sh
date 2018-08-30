
for i in {1..200}
do
    rm -r results
    mkdir results
    ./iS3D.e
    mv results oversampling/results_${i}
done 
