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
cp -r workspace/census_states package/data/census_states
cp workspace/sim_hist.csv package/data/sim_hist.csv
cp workspace/sim_ag_all.csv package/data/sim_ag_all.csv
cp workspace/tool.csv package/data/tool.csv
cp workspace/export_claims.csv package/data/export_claims.csv
cp workspace/climate.csv package/data/climate.csv
cp workspace/states.json.geojson package/data/states.json.geojson
cp workspace/states.json.qmd package/data/states.json.qmd