import unittest

import breakpoint_tasks


class CheckTasks(unittest.TestCase):
    
    def test_sample_climate_preprocess_task(self):
        self.assertIsNotNone(breakpoint_tasks.SampleClimatePreprocessTask())
    
    def test_full_climate_preprocess_task(self):
        self.assertIsNotNone(breakpoint_tasks.FullClimatePreprocessTask())
    
    def test_run_through_preprocess_task(self):
        self.assertIsNotNone(breakpoint_tasks.RunThroughPreprocessTask())
    
    def test_run_through_preprocess_future_task(self):
        self.assertIsNotNone(breakpoint_tasks.RunThroughPreprocessFutureTask())
    
    def test_run_through_sweep_task(self):
        self.assertIsNotNone(breakpoint_tasks.RunThroughSweepTask())
    
    def test_run_through_extended_sweep_task(self):
        self.assertIsNotNone(breakpoint_tasks.RunThroughExtendedSweepTask())
    
    def test_run_through_historic_project(self):
        self.assertIsNotNone(breakpoint_tasks.RunThroughHistoricProject())
    
    def test_run_through_sim_task(self):
        self.assertIsNotNone(breakpoint_tasks.RunThroughSimTask())
    
    def test_run_through_hist_task(self):
        self.assertIsNotNone(breakpoint_tasks.RunThroughHistTask())
    
    def test_execute_supplemental_tasks(self):
        self.assertIsNotNone(breakpoint_tasks.ExecuteSupplementalTasks())
    
    def test_execute_significant_long_task(self):
        self.assertIsNotNone(breakpoint_tasks.ExecuteSignificantLongTask())
    
    def test_execute_supplemental_tasks_with_cluster(self):
        self.assertIsNotNone(breakpoint_tasks.ExecuteSupplementalTasksWithCluster())
