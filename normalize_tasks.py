import csv
import json
import math

import luigi
import numpy

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
        return round(try_float(target))


def parse_row(row):
    for field in row:
        if field in const.TRAINING_STR_FIELDS:
            row[field] = row[field]
        elif field in const.TRAINING_INT_FIELDS:
            row[field] = try_int(row[field])
        else:
            row[field] = try_float(row[field])

    return row


class GetAsDeltaTaskTemplate(luigi.Task):

    def requires(self):
        return {
            'historic': preprocess_combine_tasks.CombineHistoricPreprocessTask(),
            'target': self.get_target()
        }

    def output(self):
        return luigi.LocalTarget(const.get_file_location(self.get_filename()))

    def run(self):
        averages = {}

        with self.input()['historic'].open() as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                geohash = row['geohash']

                other_fields = filter(lambda x: x not in ['year', 'geohash'], row.keys())

                for field in other_fields:
                    key = '%s.%s' % (geohash, field)
                    if key not in averages:
                        averages[key] = distribution_struct.WelfordAccumulator()

                    value = self._get_float_maybe(row[field])
                    
                    if value is not None:
                        averages[key].add(value)

        def transform_row_regular(row):
            keys = row.keys()
            keys_delta_only = filter(lambda x: x not in const.NON_DELTA_FIELDS, keys)
            keys_no_count = filter(lambda x: 'count' not in x.lower(), keys_delta_only)

            geohash = row['geohash']
            for key in keys_no_count:
                average = averages['%s.%s' % (geohash, key)].get_mean()
                delta = float(row[key]) - average
                row[key] = delta
            
            return row

        def transform_row_response(row):
            geohash = row['geohash']
            
            key = '%s.baselineYieldMean' % geohash
            original_mean = self._get_float_maybe(row['yieldMean'])
            original_std = self._get_float_maybe(row['yieldStd'])

            if original_mean is None or original_std is None or key not in averages:
                new_mean = None
                new_std = None
            else:
                baseline_mean = averages[key].get_mean()
                new_mean = (original_mean - baseline_mean) / baseline_mean
                new_std = original_std / baseline_mean

            row['yieldMean'] = new_mean
            row['yieldStd'] = new_std
            
            return row

        with self.input()['target'].open() as f_in:
            rows = csv.DictReader(f)
            rows_regular_transform = map(lambda x: transform_row_regular(x), rows)
            rows_regular_response = map(lambda x: transform_row_response(x), rows_regular_transform)

            with self.output().open('w') as f_out:
                writer = csv.DictWriter(f, fieldnames=const.TRAINING_FRAME_ATTRS)
                writer.writeheader()
                writer.writerows(rows_regular_response)

    def get_target(self):
        raise NotImplementedError('Must use implementor.')

    def get_filename(self):
        raise NotImplementedError('Must use implementor.')

    def _get_float_maybe(target):
        try:
            value = float(row[field])
        except ValueError:
            return None

        if numpy.isfinite(value):
            return None
        else:
            return value


class GetHistoricAsDeltaTask(GetAsDeltaTaskTemplate):
    
    def get_target(self):
        return preprocess_combine_tasks.CombineHistoricPreprocessTask()

    def get_filename(self):
        return 'historic_deltas_transform.csv'


class GetFutureAsDeltaTask(GetAsDeltaTaskTemplate):

    condition = luigi.Parameter()

    def get_target(self):
        return preprocess_combine_tasks.ReformatFuturePreprocessTask(condition=self.condition)

    def get_filename(self):
        return '%s_deltas_transform.csv' % self.condition


class GetInputDistributionsTask(luigi.Task):

    def requires(self):
        return GetHistoricAsDeltaTask()

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


class NormalizeTrainingFrameTemplateTask(luigi.Task):

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
            rows_with_z = map(lambda x: self._transform_z(x, distributions), rows_augmented)
            rows_with_num = map(lambda x: self._force_values(x), rows_with_z)
            rows_standardized = map(lambda x: self._standardize_row(x), rows_with_num)
            rows_with_mean = filter(
                lambda x: x['yieldMean'] != const.INVALID_VALUE,
                rows_standardized
            )
            rows_complete = filter(
                lambda x: x['yieldStd'] != const.INVALID_VALUE,
                rows_with_mean
            )

            with self.output().open('w') as f_out:
                writer = csv.DictWriter(f_out, fieldnames=const.TRAINING_FRAME_ATTRS)
                writer.writeheader()
                writer.writerows(rows_complete)

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


class NormalizeHistoricTrainingFrameTask(NormalizeTrainingFrameTemplateTask):

    def get_target(self):
        return GetHistoricAsDeltaTask()

    def get_filename(self):
        return 'historic_normalized.csv'


class NormalizeFutureTrainingFrameTask(NormalizeTrainingFrameTemplateTask):

    condition = luigi.Parameter()

    def get_target(self):
        return GetFutureAsDeltaTask(condition=self.condition)

    def get_filename(self):
        return '%s_normalized.csv' % self.condition
