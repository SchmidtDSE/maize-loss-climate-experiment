"""Optional tasks for gaussian process.

License:
    BSD
"""
import csv
import itertools
import json
import random

import luigi
import numpy
import sklearn.gaussian_process

import const
import normalize_tasks
import training_tasks

INPUT_ATTRS = training_tasks.get_input_attrs('all attrs', True)
SAMPLE_RATE = 300000


def assign_year(year):
    """
    Assign a dataset category based on the specified year.

    Args:
        year (int): The year to be evaluated.

    Returns:
        str: A string value that represents the dataset category:
             'train' if the year is before 2013,
             'test' if the year is 2013 or after and an even year,
             'valid' if the year is 2013 or after and an odd year.
    """
    if year < 2013:
        return 'train'
    else:
        return 'test' if year % 2 == 0 else 'valid'


def transform_row(target):
    """Transform a single row by assigning set category and effective year.

    Args:
        target (dict): Dictionary containing row data including year.

    Returns:
        dict: Transformed row with added setAssign and optional effectiveYear.
    """
    year = int(target['year'])

    if const.INCLUDE_YEAR_IN_MODEL:
        target['effectiveYear'] = year - 2007

    target['setAssign'] = assign_year(year)
    return target


class ResampleIndividualizeTask(luigi.Task):
    """Task that resamples and individualizes training data.
    
    This task takes normalized historic training data and creates individual samples
    based on the mean, standard deviation and sample weights. It filters rows based on
    sample weight threshold and expands them into multiple samples using Gaussian sampling.
    """

    target = luigi.Parameter()

    def requires(self):
        """Specify the dependency on normalized historic training data.

        Returns:
            NormalizeHistoricTrainingFrameTask: Task that provides normalized training data.
        """
        return normalize_tasks.NormalizeHistoricTrainingFrameTask()

    def output(self):
        """Specify the output file location for individualized samples.

        Returns:
            LocalTarget: Target for CSV file containing individualized training samples.
        """
        path = const.get_file_location('sample_individual_%s.csv' % self.target)
        return luigi.LocalTarget(path)

    def run(self):
        """Execute the resampling and individualization process.
        
        Reads normalized data, filters rows by sample weight, transforms them with
        year assignments, and expands each row into multiple samples using Gaussian
        sampling based on mean and standard deviation.
        """
        with self.input().open() as f_in:
            rows = csv.DictReader(f_in)
            transformed_rows = map(transform_row, rows)
            allowed_rows = filter(lambda x: x['setAssign'] == self.target, transformed_rows)
            expanded_rows_nested = map(lambda x: self._expand_rows(x), allowed_rows)
            expanded_rows = itertools.chain(*expanded_rows_nested)

            output_attrs = INPUT_ATTRS + ['yieldValue', 'geohash', 'year']
            with self.output().open('w') as f_out:
                writer = csv.DictWriter(f_out, fieldnames=output_attrs, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(expanded_rows)

    def _expand_rows(self, target):
        """Expand a single row into multiple samples using Gaussian sampling.

        Args:
            target (dict): Dictionary containing mean, std and sample weight information.

        Returns:
            map: Iterator of dictionaries containing individual samples with values
                drawn from Gaussian distribution.
        """
        mean = float(target['yieldMean'])
        std = float(target['yieldStd'])
        sample_weight = int(target[const.SAMPLE_WEIGHT_ATTR])
        random_array = numpy.random.random(sample_weight)
        threshold = 1 / SAMPLE_RATE
        num_samples = numpy.sum(random_array < threshold)
        samples_indexed = range(0, num_samples)

        def make_sample(index):
            value = random.gauss(mu=mean, sigma=std)
            ret_dict = dict(map(lambda x: (x, target[x]), INPUT_ATTRS))
            ret_dict['yieldValue'] = value
            return ret_dict

        return map(make_sample, samples_indexed)


class BuildGaussianProcessModelTask(luigi.Task):
    """Task that builds and trains a Gaussian Process model.
    
    This task takes the normalized training data, filters for training set rows,
    and fits a Gaussian Process Regressor with the specified kernel.
    """
    
    kernel = luigi.Parameter()
    target = luigi.Parameter()

    def requires(self):
        """Specify dependency on normalized individual instance historic training data.

        Returns:
            ResampleIndividualizeTask: Task that provides individual instance data.
        """
        return {
            'train': ResampleIndividualizeTask(target='train'),
            'summary': normalize_tasks.NormalizeHistoricTrainingFrameTask()
        }

    def output(self):
        """Specify the output file location for the trained model.

        Returns:
            LocalTarget: Target for pickle file containing trained model.
        """
        filename = 'gaussian_process_%s_eval_%s.csv' % (self.kernel, self.target)
        path = const.get_file_location(filename)
        return luigi.LocalTarget(path)

    def run(self):
        """Execute the model training process.
        
        Reads normalized data, filters for training set rows, and fits a 
        Gaussian Process model with the specified kernel.
        """
        with self.input()['train'].open() as f_in:
            rows = csv.DictReader(f_in)

            def parse_row(target):
                return {
                    'inputs': [float(target[attr]) for attr in INPUT_ATTRS],
                    'output': float(target['yieldValue'])
                }

            # Prepare inputs and outputs
            parsed_rows = [parse_row(x) for x in rows]
            inputs = [x['inputs'] for x in parsed_rows]
            outputs = [x['output'] for x in parsed_rows]

        # Train model
        model = sklearn.gaussian_process.GaussianProcessRegressor(
            kernel=self._get_kernel(self.kernel),
            copy_X_train=False,
            normalize_y=True
        )
        model.fit(inputs, outputs)

        def parse_test_row(row):
            return {
                'year': int(row['year']),
                'setAssign': row['setAssign'],
                'inputs': [float(row[attr]) for attr in INPUT_ATTRS],
                'output': {
                    'mean': float(row['yieldMean']),
                    'std': float(row['yieldStd'])
                },
                const.SAMPLE_WEIGHT_ATTR: float(row[const.SAMPLE_WEIGHT_ATTR])
            }

        def evaluate_test_row(target):
            result = model.predict([target['inputs']], return_std=True)
            return {
                'year': target['year'],
                'setAssign': target['setAssign'],
                'predictedMean': result[0][0],
                'actualMean': target['output']['mean'],
                'predictedStd': result[1][0],
                'actualStd': target['output']['std'],
                const.SAMPLE_WEIGHT_ATTR: float(target[const.SAMPLE_WEIGHT_ATTR])
            }

        # Evaluate on test
        with self.input()['summary'].open() as f_in:
            all_rows = csv.DictReader(f_in)
            all_transformed_rows = map(transform_row, all_rows)
            test_rows = filter(
                lambda x: x['setAssign'] == self.target,
                all_transformed_rows
            )
            parsed_rows = map(parse_test_row, test_rows)
            eval_rows = map(evaluate_test_row, parsed_rows)

            with self.output().open('w') as f_out:
                writer = csv.DictWriter(f_out, fieldnames=[
                    'year',
                    'setAssign',
                    'predictedMean',
                    'actualMean',
                    'predictedStd',
                    'actualStd',
                    const.SAMPLE_WEIGHT_ATTR
                ])
                writer.writeheader()
                writer.writerows(eval_rows)

    def _get_kernel(self, name):
        """
        Retrieve the kernel configuration based on the provided name.

        Args:
            name (str): The name of the kernel configuration to retrieve.

        Returns:
            kernel: The kernel setting if known, otherwise raises a NotImplementedError.

        Raises:
            NotImplementedError: If the provided kernel name is unknown.
        """
        strategies {
            'default': None
        }
        return strategies[name]


class SummarizeGaussianProcessModelTask(luigi.Task):
    """Task that summarizes Gaussian Process model performance."""
    
    kernel = luigi.Parameter()
    target = luigi.Parameter()

    def requires(self):
        """Get the tasks whose outputs are required for running the model.

        Returns:
            BuildGaussianProcessModelTask: Task that builds and evaluates the model.
        """
        return BuildGaussianProcessModelTask(
            kernel=self.kernel,
            target=self.target
        )

    def output(self):
        """Specify the output file location for the model summary.

        Returns:
            LocalTarget: Target for JSON file containing model summary metrics.
        """
        filename = 'gaussian_process_%s_summary_%s.json' % (self.kernel, self.target)
        path = const.get_file_location(filename)
        return luigi.LocalTarget(path)

    def run(self):
        """Calculate and write summary metrics."""
        with self.input().open('r') as f_in:
            rows = list(csv.DictReader(f_in))
            
            # Calculate MAE for mean and std
            def get_abs_diff(row, name):
                predicted = float(row['predicted%s' % name])
                actual = float(row['actual%s' % name])
                return abs(predicted - actual)

            mean_errors = [get_abs_diff(row, 'Mean') for row in rows]
            std_errors = [get_abs_diff(row, 'Std') for row in rows]
            
            mean_mae = sum(mean_errors) / len(mean_errors)
            std_mae = sum(std_errors) / len(std_errors)
            
            summary = {
                'mean_mae': mean_mae,
                'std_mae': std_mae,
                'kernel': self.kernel,
                'target': self.target
            }
            
            with self.output().open('w') as f_out:
                json.dump(summary, f_out, indent=2)


class SummarizeAllGaussianProcessModelTask(luigi.Task):
    """Task that summarizes Gaussian Process model performance."""

    def requires(self):
        """Varying values for sweep.

        Returns:
            Dict: Multiple values to try in sweep.
        """
        return {
            'default': SummarizeGaussianProcessModelTask(
                kernel='default',
                target='valid'
            ),
            'matern_rough': SummarizeGaussianProcessModelTask(
                kernel='matern_rough',
                target='valid'
            ),
            'matern_smooth': SummarizeGaussianProcessModelTask(
                kernel='matern_smooth',
                target='valid'
            ),
            'matern_rough_white': SummarizeGaussianProcessModelTask(
                kernel='matern_rough_white',
                target='valid'
            ),
            'matern_rough_white': SummarizeGaussianProcessModelTask(
                kernel='matern_rough_white',
                target='valid'
            )
        }

    def output(self):
        """Unified JSON document describing the sweep.

        Returns:
            LocalTarget: Target for JSON file containing model summary metrics.
        """
        filename = 'gaussian_process_all_summary_valid.json'
        path = const.get_file_location(filename)
        return luigi.LocalTarget(path)

    def run(self):
        """Calculate and write summary metrics."""

        ret_dict = {}
        
        def get_json(name):
            with self.input()[name].open() as f:
                component = json.load(f)
                ret_dict[name] = component

        get_json('default')
        get_json('matern_rough')
        get_json('matern_smooth')
        get_json('matern_rough_white')
        get_json('matern_rough_white')

        with self.output().open('w') as f_out:
            json.dump(ret_dict, f_out, indent=2)
