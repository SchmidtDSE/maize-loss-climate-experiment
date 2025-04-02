python3 hist_viz.py ../outputs/export_hist.csv 2050 75 unit counterfactual ../img/hist.png
python3 results_viz_entry.py ../outputs/export_summary.csv 2050 risk neighborhood 0.05 Bonferroni all none jul 75 ../outputs/export_climate.csv ../img/map.png
python3 results_viz_entry.py ../outputs/export_summary.csv 2050 risk scatter 0.05 Bonferroni significant chirps jul 75 ../outputs/export_climate.csv ../img/scatter.png
python3 std_hist.py ../outputs/export_combined_tasks.csv ../img/std.png
python3 history_viz.py ../outputs/export_claims.csv ../img/yield_sim.png std high
