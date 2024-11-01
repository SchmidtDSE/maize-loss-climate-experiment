echo "Cleaning post-data preprocessing steps."

rm -r clean_hold
mkdir clean_hold
mv ./workspace/climate_historic.csv ./clean_hold
mv ./workspace/climate_2030_SSP245.csv ./clean_hold
mv ./workspace/climate_2050_SSP245.csv ./clean_hold
mv ./workspace/yield.csv ./clean_hold
mv ./workspace/unit_sizes_2023.csv ./clean_hold

rm -r workspace
mv clean_hold workspace

