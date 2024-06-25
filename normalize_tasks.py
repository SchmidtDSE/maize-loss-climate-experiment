import csv
import json
import math

import luigi

import const
import distribution_struct
import preprocess_combine_tasks


def try_float(target):
    try:
        return float(target)
    except ValueError:
        return None


def try_int(target):
    try:
        return int(target)
    except ValueError:
        return None


def parse_row(row):
    for field in row:
        if field in const.TRAINING_STR_FIELDS:
            row[field] = row[field]
        elif field in const.TRAINING_INT_FIELDS:
            row[field] = try_int(row[field])
        else:
            row[field] = try_float(row[field])

    return row


class GetInputDistributionsTask(luigi.Task):

    def requires(self):
        return preprocess_combine_tasks.CombineHistoricPreprocessTask()

    def output(self):
        return luigi.LocalTarget(const.get_file_location('historic_z.csv'))

    def run(self):
        fields_to_process = set(const.TRAINING_FRAME_ATTRS) - set(const.NON_Z_FIELDS)
        accumulators = dict(map(
            lambda x: (x, distribution_struct.WelfordAccumulator()),
            fields_to_process
        ))

        with self.input().open('r') as f:
            input_records = csv.DictReader(f)

            for row in input_records:
                for field in fields_to_process:
                    value = try_float(row[field])
                    if value is not None and not math.isnan(value):
                        accumulators[field].add(value)

        output_rows = map(
            lambda x: self._serialize_accumulator(x[0], x[1]),
            accumulators.items()
        )

        with self.output().open('w') as f:
            writer = csv.DictWriter(f, fieldnames=['field', 'mean', 'std'])
            writer.writeheader()
            writer.writerows(output_rows)

    def _serialize_accumulator(self, field, accumulator):
        return {
            'field': field,
            'mean': accumulator.get_mean(),
            'std': accumulator.get_std()
        }


class NormalizeTrainingFrameTask(luigi.Task):

    def requires(self):
        return {
            'distributions': GetInputDistributionsTask(),
            'target': self.get_target()
        }

    def output(self):
        return luigi.LocalTarget(const.get_file_location(self.get_filename()))

    def run(self):
        with self.input()['distributions'].open('r') as f:
            rows = csv.DictReader(f)

            distributions = {}

            for row in rows:
                distributions[row['field']] = {
                    'mean': float(row['mean']),
                    'std': float(row['std'])
                }

        with self.input()['target'].open('r') as f_in:
            reader = csv.DictReader(f_in)

            rows = map(lambda x: self._parse_row(x), reader)
            rows_allowed = filter(lambda x: x['year'] in const.YEARS, rows)
            rows_augmented = map(lambda x: self._set_aside_attrs(x), rows_allowed)
            rows_with_response = map(lambda x: self._transform_yield(x, distributions), rows_augmented)
            rows_with_z = map(lambda x: self._transform_z(x, distributions), rows_with_response)
            rows_with_num = map(lambda x: self._force_values(x), rows_with_z)
            rows_standardized = map(lambda x: self._standardize_row(x), rows_with_num)

            with self.output().open('w') as f_out:
                writer = csv.DictWriter(f_out, fieldnames=const.TRAINING_FRAME_ATTRS)
                writer.writeheader()
                writer.writerows(rows_standardized)

    def get_target(self):
        raise NotImplementedError('Use implementor.')

    def get_filename(self):
        raise NotImplementedError('Use implementor.')

    def _parse_row(self, row):
        return parse_row(row)

    def _set_aside_attrs(self, row):
        row['baselineYieldMeanOriginal'] = row['baselineYieldMean']
        row['baselineYieldStdOriginal'] = row['baselineYieldStd']
        return row

    def _transform_yield(self, row, distributions):
        distribution = distributions['baselineYieldMean']
        for field in const.YIELD_FIELDS:
            row[field] = (row[field] - distribution['mean']) / distribution['std']

        return row

    def _transform_z(self, row, distributions):
        fields = distributions.keys()
        fields_allowed = filter(lambda x: x not in const.YIELD_FIELDS, fields)
        for field in fields_allowed:
            original_value = row[field]
            
            distribution = distributions[field]
            mean = distribution['mean']
            std = distribution['std']

            original_value = row[field]

            all_zeros = original_value == 0 and mean == 0 and std == 0

            if (original_value is None) or all_zeros:
                row[field] = const.INVALID_VALUE
            else:
                row[field] = (original_value - mean) / std

        return row

    def _force_values(self, row):
        def force_value(target):
            if target is None or math.isnan(target):
                return const.INVALID_VALUE
            else:
                return target

        for field in filter(lambda x: x != 'geohash', row.keys()):
            row[field] = force_value(row[field])

        return row

    def _standardize_row(self, row):
        return dict(map(lambda x: (x, row[x]), const.TRAINING_FRAME_ATTRS))


class NormalizeHistoricTrainingFrameAbsTask(NormalizeTrainingFrameTask):

    def get_target(self):
        return preprocess_combine_tasks.CombineHistoricPreprocessTask()

    def get_filename(self):
        return 'historic_normalized.csv'
