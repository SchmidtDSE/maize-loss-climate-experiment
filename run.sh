mkdir workspace

python3 -m luigi --module breakpoint_tasks RunThroughPreprocessTask --local-scheduler --workers 8
python3 -m luigi --module breakpoint_tasks ExecuteAllWithCluster --local-scheduler --workers 4
