"""Tasks to build optional statistics describing model performance.

License: BSD
"""
import csv
import json

import luigi

import preprocess_combine_tasks

BIOMASS_CONVERSION_FACTOR = 1 / 50


class GetTestErrorTonnesPerHectareTask(luigi.Task):
    """Task to calculate test set average absolute error in tonnes per hectare."""

    def requires(self):
        """Return required tasks that provide test data and average yield.

        Returns:
            Dict containing required tasks
        """
        return {
            'average': GetAverageTestTonnesPerHectareTask(),
            'residuals': selection_tasks.PostHocTestRawDataRetrainCountTask()
        }

    def output(self):
        """Determine where the error should be written.

        Returns:
            LocalTarget where the error should be written.
        """
        return luigi.LocalTarget('test_error_tonnes.json')

    def run(self):
        """Calculate the test set average absolute error in tonnes per hectare."""
        # Get average test tonnes per hectare
        with self.input()['average'].open('r') as f:
            average_data = json.load(f)
            average_tonnes = average_data['tonnes_per_hectare']

        # Calculate weighted average of absolute residuals
        total_weighted_residual = 0.0
        total_count = 0

        with self.input()['residuals'].open('r') as f:
            reader = csv.DictReader(f)
            test_records = filter(lambda x: x['setAssign'] == 'test', reader)

            for row in test_records:
                residual = abs(float(row['meanResidual']))
                count = float(row['yieldObservations'])
                
                total_weighted_residual += residual * count
                total_count += count

        avg_abs_residual = total_weighted_residual / total_count if total_count > 0 else 0
        error_tonnes = average_tonnes * avg_abs_residual

        with self.output().open('w') as f:
            json.dump({'error_tonnes_per_hectare': error_tonnes}, f)


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
            test_records = filter(lambda x: int(x['year']) in [2013, 2015], reader)

            for row in test_records:
                mean = float(row['yieldMean'])
                count = float(row['yieldObservations'])
                
                total_weighted_sum += mean * count
                total_count += count

        biomass = total_weighted_sum / total_count if total_count > 0 else 0
        tonnes_per_hectare = biomass * BIOMASS_CONVERSION_FACTOR

        with self.output().open('w') as f:
            json.dump({'tonnes_per_hectare': tonnes_per_hectare}, f)



