mkdir package

mkdir package/outputs
cp workspace/stats.json package/outputs/stats.json
cp workspace/export_sweep.csv package/outputs/export_sweep.csv
cp workspace/export_hist.csv package/outputs/export_hist.csv
cp workspace/export_summary.csv package/outputs/export_summary.csv
cp workspace/export_claims.csv package/outputs/export_claims.csv
cp workspace/export_combined_tasks.csv package/outputs/export_combined_tasks.csv
cp workspace/export_climate.csv package/outputs/export_climate.csv

mkdir package/data
cp workspace/export_hist.csv package/data/sim_hist.csv
cp workspace/export_sweep.csv package/data/sim_ag_all.csv
cp workspace/export_summary.csv package/data/tool.csv
cp workspace/export_claims.csv package/data/export_claims.csv
cp workspace/export_climate.csv package/data/climate.csv

rm -r package/build
mkdir package/build
cd package/build
wget https://ag-adaptation-study.pub/archive/data.zip
unzip data.zip
cd ../..

cp -r package/build/data/census_states package/data/census_states
cp package/build/data/states.json.geojson package/data/states.json.geojson
cp package/build/data/states.json.qmd package/data/states.json.qmd

cd package
rm data.zip
rm outputs.zip
zip data.zip -r data
zip outputs.zip -r outputs