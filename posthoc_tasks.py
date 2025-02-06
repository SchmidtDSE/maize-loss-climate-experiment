"""Optional tasks for model comparison and contextualization.

License:
    BSD
"""
import csv
import random

import luigi

import const
import normalize_tasks
import training_tasks

INPUT_ATTRS = training_tasks.get_input_attrs([], True)
SAMPLE_RATE = 1000


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


class ResampleIndividualizeTask(luigi.Task):

    def requires(self):
        return normalize_tasks.NormalizeHistoricTrainingFrameTask()

    def output(self):
        return luigi.LocalTarget(const.get_file_location('train_sample_individual.csv'))

    def run(self):

        with self.input().open() as f_in:
            rows = csv.DictReader(f_in)
            allowed_rows = filter(lambda x: int(x[const.SAMPLE_WEIGHT_ATTR]) > SAMPLE_RATE, rows)
            transformed_rows = map(lambda x: self._transform_row(x), allowed_rows)
            expanded_rows_nested = map(lambda x: self._expand_rows(x), transformed_rows)
            expanded_rows = itertools.chain(*expanded_rows_nested)

            output_attrs = INPUT_ATTRS + ['setAssign', 'value']
            with self.output().open('w') as f_out:
                writer = csv.DictWriter(f_out, fieldnames=output_attrs)
                writer.writeheader()
                writer.writerows(expanded_rows)

    def _transform_row(self, target):
        year = int(target['year'])
    
        if const.INCLUDE_YEAR_IN_MODEL:
            target['effectiveYear'] = year - 2007
    
        target['setAssign'] = assign_year(year)
        return target

    def _expand_rows(self, target):
        num_samples = round(target[const.SAMPLE_WEIGHT_ATTR] / SAMPLE_RATE)
        mean = float(target['mean'])
        std = float(target['std'])
        samples_indexed = range(0, num_samples)

        def make_sample(index):
            value = random.gauss(mu=mean, sigma=std)
            ret_dict = dict(map(lambda x: (x, target[x]), INPUT_ATTRS))
            ret_dict['value'] = value
            return ret_dict

        return map(make_sample, samples_indexed)


class 
