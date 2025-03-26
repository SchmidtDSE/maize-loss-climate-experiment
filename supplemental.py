
import csv
import json

import luigi

import preprocess_combine_tasks

class GetAverageTestTonnesPerHectareTask(luigi.Task):
    """Task to calculate average test tonnes per hectare from historic data."""

    def requires(self):
        """Return the required task which provides the historic data.

        Returns:
            CombineHistoricPreprocessTask
        """
        return preprocess_combine_tasks.CombineHistoricPreprocessTask()

    def output(self):
        """Determine where the average should be written.

        Returns:
            LocalTarget where the average should be written.
        """
        return luigi.LocalTarget('average_test_tonnes.json')

    def run(self):
        """Calculate the weighted average and convert to tonnes per hectare."""
        total_weighted_sum = 0.0
        total_count = 0

        with self.input().open() as f:
            reader = csv.DictReader(f)

            for row in reader:
                year = int(row['year'])
                
                if year in [2013, 2015]:
                    mean = float(row['yieldMean'])
                    count = float(row['yieldObservations'])
                    
                    total_weighted_sum += mean * count
                    total_count += count

        biomass = total_weighted_sum / total_count if total_count > 0 else 0
        tonnes_per_hectare = biomass / 50

        with self.output().open('w') as f:
            json.dump({'tonnes_per_hectare': tonnes_per_hectare}, f)
