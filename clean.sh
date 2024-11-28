echo "Cleaning post-data preprocessing steps."

rm -r clean_hold
mkdir clean_hold

mv ./workspace/climate_historic.csv ./clean_hold
mv ./workspace/climate_2030_SSP245.csv ./clean_hold
mv ./workspace/climate_2050_SSP245.csv ./clean_hold
mv ./workspace/yield.csv ./clean_hold
mv ./workspace/yield_beta.csv ./clean_hold
mv ./workspace/unit_sizes_2023.csv ./clean_hold
mv ./workspace/usda_post_summary.csv ./clean_hold

# mv ./workspace/geohash_shapes.csv ./clean_hold
# mv ./workspace/historic_averages.csv ./clean_hold
# mv ./workspace/historic_deltas_transform.csv ./clean_hold
# mv ./workspace/training_frame.csv ./clean_hold

rm -r workspace
mv clean_hold workspace

