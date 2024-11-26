"""Tasks to help standardize / normalize data prior to models.

License:
    BSD
"""
import csv
import math

import luigi
import more_itertools
import numpy
import scipy.stats

import const
import distribution_struct
import parse_util
import preprocess_combine_tasks


def parse_row(row):
    """Parse all fields in an input training row.

    Args:
        row: The row for which fields should be parsed.

    Returns:
        Row after parsing.
    """
    for field in row:
        if field in const.TRAINING_STR_FIELDS:
            row[field] = row[field]
        elif field in const.TRAINING_INT_FIELDS:
            row[field] = parse_util.try_int(row[field])
        else:
            row[field] = parse_util.try_float(row[field])

    return row


def get_finite_maybe(target):
    """Force a numeric value to be finte.

    Try converting a value to a finite float and return None if parsing fails or the value is
    infinite.

    Args:
        target: The string to parse.

    Returns:
        The number parsed.
    """
    value = parse_util.try_float(target)

    if value is not None and numpy.isfinite(value):
        return value
    else:
        return None


def transform_row_response(task, make_imports=False, response_available=True):
    row = task['row']
    baseline_mean = task['baseline_mean']
    baseline_std = task['baseline_std']
    target_a = task['target_a']
    target_b = task['target_b']
    target_loc = task['target_loc']
    target_scale = task['target_scale']

    values_required = [target_a, target_b, target_loc, target_scale]
    values_none = filter(lambda x: x is None, values_required)
    num_values_none = sum(map(lambda x: 1, values_none))
    values_not_given = num_values_none > 0
    baseline_not_given = baseline_mean is None or baseline_std is None

    complete = not (values_not_given or baseline_not_given)

    if complete and response_available:
        target_dist = scipy.stats.beta.rvs(
            target_a,
            target_b,
            loc=target_loc,
            scale=target_scale,
            size=5000
        )

        deltas = (target_dist - baseline_mean) / baseline_mean
        deltas_ln = numpy.arcsinh(deltas)
        new_mean = numpy.mean(deltas)
        new_std = numpy.std(deltas)
        new_skew = scipy.stats.skew(deltas_ln)
        new_kurtosis = scipy.stats.kurtosis(deltas_ln)
    else:
        new_mean = None
        new_std = None
        new_skew = None
        new_kurtosis = None

    row['yieldMean'] = new_mean
    row['yieldStd'] = new_std
    row['skewLn'] = new_skew
    row['kurtosisLn'] = new_kurtosis
    row['isComplete'] = complete

    return row


class GetHistoricAveragesTask(luigi.Task):
    """Get the historic average yield at the geohash-level."""

    def requires(self):
        """Return the required task which normalizes historic data.

        Returns:
            CombineHistoricPreprocessTask
        """
        return preprocess_combine_tasks.CombineHistoricPreprocessTask()

    def output(self):
        """Determine where the averages should be written.

        Returns:
            LocalTarget where the averages should be written.
        """
        return luigi.LocalTarget(const.get_file_location('historic_averages.csv'))

    def run(self):
        """Compute the averages."""
        averages = {}

        with self.input().open() as f:
            reader = csv.DictReader(f)

            for row in reader:
                geohash = row['geohash']

                other_fields = filter(lambda x: x not in ['year', 'geohash'], row.keys())

                for field in other_fields:
                    key = '%s.%s' % (geohash, field)
                    if key not in averages:
                        averages[key] = distribution_struct.WelfordAccumulator()

                    value = get_finite_maybe(row[field])

                    if value is not None:
                        averages[key].add(value)

        output_rows = map(lambda x: {'key': x[0], 'mean': x[1].get_mean()}, averages.items())

        with self.output().open('w') as f:
            writer = csv.DictWriter(f, fieldnames=['key', 'mean'])
            writer.writeheader()
            writer.writerows(output_rows)


class GetAsDeltaTaskTemplate(luigi.Task):
    """Template for task which converts to yield deltas.

    Abstract base class which serves as a template for task which convert from yields to yield
    deltas.
    """

    def requires(self):
        """Require that the averages and the dataset to convert needs to be available.

        Returns:
            The GetHistoricAveragesTask which provides averages and the target to convert.
        """
        return {
            'averages': GetHistoricAveragesTask(),
            'target': self.get_target()
        }

    def output(self):
        """Determine where the deltas should be written.

        Returns:
            LocalTarget where the updated data should be written.
        """
        return luigi.LocalTarget(const.get_file_location(self.get_filename()))

    def run(self):
        """Convert from yields to yield deltas."""

        with self.input()['averages'].open() as f:
            reader = csv.DictReader(f)
            average_tuples_str = map(lambda x: (x['key'], x['mean']), reader)
            average_tuples = map(lambda x: (x[0], float(x[1])), average_tuples_str)
            averages = dict(average_tuples)

        def transform_row_regular(row, averages):
            keys = row.keys()
            keys_delta_only = filter(lambda x: x not in const.NON_DELTA_FIELDS, keys)
            keys_no_count = filter(lambda x: 'count' not in x.lower(), keys_delta_only)

            geohash = row['geohash']
            for key in keys_no_count:
                average = averages['%s.%s' % (geohash, key)]
                original_value = get_finite_maybe(row[key])
                if original_value is not None:
                    delta = original_value - average
                    row[key] = delta

            return row

        def make_task(row, averages):
            geohash = row['geohash']

            mean_key = '%s.baselineYieldMean' % geohash
            std_key = '%s.baselineYieldStd' % geohash

            baseline_mean = averages[mean_key]
            baseline_std = averages[std_key]

            target_mean = get_finite_maybe(row['yieldMean'])
            target_std = get_finite_maybe(row['yieldStd'])
            target_a = get_finite_maybe(row['yieldA'])
            target_b = get_finite_maybe(row['yieldB'])
            target_loc = get_finite_maybe(row['yieldLoc'])
            target_scale = get_finite_maybe(row['yieldScale'])

            return {
                'row': row,
                'baseline_mean': baseline_mean,
                'baseline_std': baseline_std,
                'target_mean': target_mean,
                'target_std': target_std,
                'target_a': target_a,
                'target_b': target_b,
                'target_loc': target_loc,
                'target_scale': target_scale
            }

        response_available = self._get_response_available()
        with self.input()['target'].open() as f_in:
            rows = csv.DictReader(f_in)

            rows_regular_transform = map(
                lambda x: transform_row_regular(
                    x,
                    averages
                ),
                rows
            )

            rows_regular_response_tasks = map(
                lambda x: make_task(
                    x,
                    averages
                ),
                rows_regular_transform
            )

            # TODO: This could be made distributed if later desired.
            futures = map(
                lambda x: transform_row_response(x, response_available=response_available),
                rows_regular_response_tasks
            )
            futures_chunked_unrealized = more_itertools.ichunked(futures, 200)
            futures_chunked = map(lambda x: list(x), futures_chunked_unrealized)

            with self.output().open('w') as f_out:
                writer = csv.DictWriter(
                    f_out,
                    fieldnames=const.TRAINING_FRAME_ATTRS,
                    extrasaction='ignore'
                )
                writer.writeheader()

                def is_approx_normal(target):
                    if not response_available:
                        return True

                    if target['skewLn'] is None or abs(target['skewLn']) > 2:
                        return False

                    if target['kurtosisLn'] is None or abs(target['kurtosisLn']) > 7:
                        return False

                    return True

                total_count = 0
                complete_count = 0
                normal_count = 0
                for chunk in futures_chunked:
                    # If using distribution: for row in map(lambda x: x.result(), chunk):
                    for row in chunk:
                        total_count += 1
                        complete_count += 1 if row['isComplete'] else 0
                        normal_count += 1 if (row['isComplete'] and is_approx_normal(row)) else 0
                        writer.writerow(row)

                if response_available:
                    assert complete_count > 0

                    complete_rate = complete_count / total_count
                    normality_rate = normal_count / complete_count

                    debug_loc = const.get_file_location(self.get_filename() + '-norm.txt')
                    with open(debug_loc, 'w') as f:
                        f.write('Normal: %f. Complete: %f.' % (normality_rate, complete_rate))

                    if complete_rate < 0.95:
                        raise RuntimeError(
                            'Complete rate: %f' % complete_rate
                        )

                    if normality_rate < 0.95:
                        raise RuntimeError(
                            'Normality assumption rate: %f' % normality_rate
                        )

    def get_target(self):
        """Get the task whose output should be converted to yield deltas.

        Returns:
            Luigi task.
        """
        raise NotImplementedError('Must use implementor.')

    def get_filename(self):
        """Get the filename to which the results should be written.

        Returns:
            Filename as string (not path).
        """
        raise NotImplementedError('Must use implementor.')


class GetHistoricAsDeltaTask(GetAsDeltaTaskTemplate):
    """Convert historic data to yield deltas."""

    def get_target(self):
        """Get the task whose output should be converted to yield deltas.

        Returns:
            Luigi task.
        """
        return preprocess_combine_tasks.CombineHistoricPreprocessBetaTask()

    def get_filename(self):
        """Get the filename to which the results should be written.

        Returns:
            Filename as string (not path).
        """
        return 'historic_deltas_transform.csv'
    
    def _get_response_available(self):
        return True


class GetFutureAsDeltaTask(GetAsDeltaTaskTemplate):
    """Convert a future dataset to yield deltas."""

    condition = luigi.Parameter()

    def get_target(self):
        """Get the task whose output should be converted to yield deltas.

        Returns:
            Luigi task.
        """
        return preprocess_combine_tasks.ReformatFuturePreprocessBetaTask(condition=self.condition)

    def get_filename(self):
        """Get the filename to which the results should be written.

        Returns:
            Filename as string (not path).
        """
        return '%s_deltas_transform.csv' % self.condition
    
    def _get_response_available(self):
        return False


class GetInputDistributionsTask(luigi.Task):
    """Task to get overall variable distributions needed for z score normalization."""

    def requires(self):
        """Get the task which provides yield deltas and input variables.

        Returns:
            GetHistoricAsDeltaTask
        """
        return GetHistoricAsDeltaTask()

    def output(self):
        """Get the location at which information needed for z score normalization should be written.

        Returns:
            LocalTarget at which distributional information should be written as required for z
            scores.
        """
        return luigi.LocalTarget(const.get_file_location('historic_z.csv'))

    def run(self):
        """Find the distributional information for input variables."""
        fields_to_process = set(const.TRAINING_FRAME_ATTRS) - set(const.NON_Z_FIELDS)
        accumulators = dict(map(
            lambda x: (x, distribution_struct.WelfordAccumulator()),
            fields_to_process
        ))

        with self.input().open('r') as f:
            input_records = csv.DictReader(f)

            for row in input_records:
                for field in fields_to_process:
                    value = parse_util.try_float(row[field])
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
        """Serialize the results of an accumulator to a primitives only dictionary.

        Args:
            field: The field for which distributional information should be serialized.
            accumulator: The accumulator for the given field.

        Returns:
            Serialized accumulator for a single field.
        """
        return {
            'field': field,
            'mean': accumulator.get_mean(),
            'std': accumulator.get_std()
        }


class NormalizeTrainingFrameTemplateTask(luigi.Task):
    """Template for task which normalizes inputs for use with the neural network.

    Abstract base class (template class) for task which normalizes inputs for use with the neural
    network or other statistical tasks downstream dependent on that standardized frame.
    """

    def requires(self):
        """Indicate that distributional data is needed for z scores and indicate target.

        Returns:
            GetInputDistributionsTask and the get_target() to normalize.
        """
        return {
            'distributions': GetInputDistributionsTask(),
            'target': self.get_target()
        }

    def output(self):
        """Indicate where the normalized frame should be written.

        Returns:
            LocalTarget where the normalized frame should be written.
        """
        return luigi.LocalTarget(const.get_file_location(self.get_filename()))

    def run(self):
        """Execute normalization."""
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
            rows_standardized = map(lambda x: self._standardize_fields(x), rows_with_num)

            if self._require_response():
                rows_with_mean = filter(
                    lambda x: x['yieldMean'] != const.INVALID_VALUE,
                    rows_standardized
                )
                rows_complete = filter(
                    lambda x: x['yieldStd'] != const.INVALID_VALUE,
                    rows_with_mean
                )
                rows_to_write = rows_complete
            else:
                rows_to_write = rows_standardized

            with self.output().open('w') as f_out:
                writer = csv.DictWriter(f_out, fieldnames=const.TRAINING_FRAME_ATTRS)
                writer.writeheader()
                writer.writerows(rows_to_write)

    def get_target(self):
        """Get the task whose output should be normalized.

        Returns:
            Luigi task
        """
        raise NotImplementedError('Use implementor.')

    def get_filename(self):
        """Get the filename where the output should be written within workspace.

        Returns:
            String filename (not full path).
        """
        raise NotImplementedError('Use implementor.')

    def _parse_row(self, row):
        """Parse an input row, converting strings to numbers were appropriate.

        Args:
            row: The row to parse.

        Returns:
            The row after parsing.
        """
        return parse_row(row)

    def _set_aside_attrs(self, row):
        """Make copies of some values prior to normalization for informational purposes.

        Some fields' original values can be useful for statistical purposes downstream and this
        step makes a copy prior to normalization.

        Args:
            row: The row to modify.

        Returns:
            The row after normlaization.
        """
        row['baselineYieldMeanOriginal'] = row['baselineYieldMean']
        row['baselineYieldStdOriginal'] = row['baselineYieldStd']
        return row

    def _transform_z(self, row, distributions):
        """Transform fields to z scores.

        Args:
            row: The row to transform.
            distributions: Distributional information about each field needed for z normalization.

        Returns:
            Row after normalization.
        """
        fields = distributions.keys()

        if const.NORM_YIELD_FIELDS:
            fields_allowed = fields
        else:
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
        """Ensure that all numeric values are valid (no None or NaN).

        Ensure that all numeric values are valid (no None or NaN), using the sentinel INVALID_VALUE
        where needed.

        Args:
            row: The row whose values should be ensured.

        Returns:
            The row after ensuring numeric values.
        """
        def force_value(target):
            if target is None or math.isnan(target):
                return const.INVALID_VALUE
            else:
                return target

        for field in filter(lambda x: x != 'geohash', row.keys()):
            row[field] = force_value(row[field])

        return row

    def _standardize_fields(self, row):
        """Check that expected and only expected fields are present.

        Args:
            row: The dictionary to check for expected fields.

        Returns:
            The row with only expected fields.
        """
        return dict(map(lambda x: (x, row[x]), const.TRAINING_FRAME_ATTRS))
    
    def _require_response(self):
        raise NotImplementedError('Use implementor.')


class NormalizeHistoricTrainingFrameTask(NormalizeTrainingFrameTemplateTask):
    """Task to normalize historic actuals."""

    def get_target(self):
        """Indicate that this task should normalize historic data with yield deltas.

        Returns:
            GetHistoricAsDeltaTask
        """
        return GetHistoricAsDeltaTask()

    def get_filename(self):
        """Get the filename where the outputs should be written.

        Returns:
            historic_normalized.csv
        """
        return 'historic_normalized.csv'
    
    def _require_response(self):
        return True


class NormalizeFutureTrainingFrameTask(NormalizeTrainingFrameTemplateTask):
    """Task which normalizes a future data series, using condition parameter."""

    condition = luigi.Parameter()  # Expects value like 2050_SSP245.

    def get_target(self):
        """
        Indicate that this task should normalize future data with yield deltas.

        Returns:
            GetFutureAsDeltaTask
        """
        return GetFutureAsDeltaTask(condition=self.condition)

    def get_filename(self):
        """Get the filename where the outputs should be written.

        Returns:
            String filename, not full path.
        """
        return '%s_normalized.csv' % self.condition
    
    def _require_response(self):
        return False
