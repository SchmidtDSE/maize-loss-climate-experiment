"""Tasks which generate the data artifacts needed to run the tools and fill in the paper template.

License:
    BSD
"""

import csv
import itertools
import json
import math

import geolib.geohash
import luigi
import toolz.itertoolz

import const
import distribution_struct
import normalize_tasks
import sim_tasks
import training_tasks

CLIMATE_OUTPUT_COLS = [
    'year',
    'geohash',
    'month',
    'rhnMeanChange',
    'rhxMeanChange',
    'tmaxMeanChange',
    'tminMeanChange',
    'chirpsMeanChange',
    'svpMeanChange',
    'vpdMeanChange',
    'wbgtmaxMeanChange'
]

CLIMATE_ATTRS = [
    'rhn',
    'rhx',
    'tmax',
    'tmin',
    'chirps',
    'svp',
    'vpd',
    'wbgtmax'
]

MONTHS = list(range(1, 13))

SWEEP_OUTPUT_COLS = [
    'block',
    'layers',
    'l2Reg',
    'dropout',
    'allowCount',
    'trainMean',
    'trainStd',
    'validMean',
    'validStd',
    'testMean',
    'testStd'
]

HIST_OUTPUT_COLS = ['geohashSize', 'set', 'series', 'bin', 'val']

TOOL_OUTPUT_COLS = [
    'geohash',
    'year',
    'condition',
    'lossThreshold',
    'num',
    'predictedRisk',
    'counterfactualRisk',
    'p',
    'pMeets0.05n',
    'pMeets0.10n',
    'predictedMean',
    'counterfactualMean',
    'lat',
    'lng'
]

COMBINED_TASK_FIELDS = [
    'geohash',
    'year',
    'condition',
    'originalYieldMean',
    'originalYieldStd',
    'projectedYieldMean',
    'projectedYieldStd',
    'numObservations'
]

CLAIMS_COLS = [
    'offsetBaseline',
    'year',
    'condition',
    'threshold',
    'thresholdStd',
    'stdMult',
    'geohashSimSize',
    'claimsRate',
    'claimsRateStd',
    'num'
]

CLAIMS_RATE_GROUP_KEYS = [
    'offsetBaseline',
    'year',
    'condition',
    'threshold',
    'thresholdStd',
    'stdMult',
    'geohashSimSize'
]

CLAIMS_RATE_GROUP_KEYS_FLOAT = [
    'threshold',
    'thresholdStd',
    'stdMult',
    'geohashSimSize',
    'num',
    'claimsRate',
    'claimsRateStd'
]

USE_UNIT_FOR_COUNTERFACTUAL = True


class ClimateExportTask(luigi.Task):
    """Task which exports a summary of climate variables in historic as well as predicted series."""

    def requires(self):
        """Require that the climate data go through normalization for all series.

        Returns:
            Tasks required to run this task.
        """
        return {
            'historic': normalize_tasks.NormalizeHistoricTrainingFrameTask(),
            '2030': normalize_tasks.NormalizeFutureTrainingFrameTask(condition='2030_SSP245'),
            '2050': normalize_tasks.NormalizeFutureTrainingFrameTask(condition='2050_SSP245')
        }

    def output(self):
        """Indicate output location where the climate variable summaries will be written.

        Returns:
            LocalTarget for CSV file.
        """
        return luigi.LocalTarget(const.get_file_location('export_climate.csv'))

    def run(self):
        """Summarize the input climate data as a simplified CSV output."""
        historic_accumulators = {}
        with self.input()['historic'].open() as f:
            reader = csv.DictReader(f)

            for row in reader:
                pieces = itertools.product(CLIMATE_ATTRS, MONTHS)
                for attr, month in pieces:
                    row_key = '%sMean%d' % (attr, month)
                    value_str = row[row_key]

                    if value_str != '-999':
                        value = float(value_str)

                        if attr not in historic_accumulators:
                            historic_accumulators[attr] = distribution_struct.WelfordAccumulator()

                        historic_accumulators[attr].add(value)

        output_rows = itertools.chain(
            self._build_export_for_year(2030, historic_accumulators),
            self._build_export_for_year(2050, historic_accumulators)
        )

        with self.output().open('w') as f:
            writer = csv.DictWriter(f, fieldnames=CLIMATE_OUTPUT_COLS)
            writer.writeheader()
            writer.writerows(output_rows)

    def _build_export_for_year(self, year, historic_accumulators):
        """Build a climate export for a single year.

        Args:
            year: The year to summarize.
            historic_accumulators: Accumulators with data for the given year.

        Returns:
            Primitives-only dictionary summarizing all climate variables for a year.
        """
        accumulators = {}
        with self.input()[str(year)].open() as f:
            reader = csv.DictReader(f)

            for row in reader:
                geohash = row['geohash']

                pieces = itertools.product(CLIMATE_ATTRS, MONTHS)
                for attr, month in pieces:
                    row_key = '%sMean%d' % (attr, month)

                    mean = historic_accumulators[attr].get_mean()
                    std = historic_accumulators[attr].get_std()
                    value_str = row[row_key]

                    if value_str != '-999':
                        value = (float(row[row_key]) - mean) / std

                        accumulator_key = self._get_accumulator_key(geohash, month, attr)
                        if accumulator_key not in accumulators:
                            accumulators[accumulator_key] = distribution_struct.WelfordAccumulator()

                        accumulator = accumulators[accumulator_key]
                        accumulator.add(value)

        accumulator_keys = accumulators.keys()

        def get_accumulator_dict(key):
            target_dict = self._parse_accumulator_key(key)
            target_dict['value'] = accumulators[key].get_mean()
            return target_dict

        accumulator_dicts = map(get_accumulator_dict, accumulator_keys)

        def make_output_piece(input_dict):
            output_dict = {
                'year': year,
                'geohash': input_dict['geohash'],
                'month': input_dict['month']
            }
            output_dict['%sMeanChange' % input_dict['attr']] = input_dict['value']
            return output_dict

        output_pieces_dict = map(make_output_piece, accumulator_dicts)

        def combine_output_pieces(a, b):
            output_dict = {
                'year': a['year'],
                'geohash': a['geohash'],
                'month': a['month']
            }

            a_keys = set(a.keys())
            b_keys = set(b.keys())
            keys_to_combine = (a_keys.union(b_keys)) - {'year', 'geohash', 'month'}

            for key in keys_to_combine:
                if key in a:
                    output_dict[key] = a[key]
                else:
                    output_dict[key] = b[key]

            return output_dict

        def get_combine_key(target):
            pieces = [target['year'], target['month'], target['geohash']]
            pieces_str = map(lambda x: str(x), pieces)
            return '\t'.join(pieces_str)

        output_rows = toolz.itertoolz.reduceby(
            get_combine_key,
            combine_output_pieces,
            output_pieces_dict
        ).values()

        def is_output_valid(target):
            output_keys = set(target.keys()) - {'year', 'geohash', 'month'}
            output_values = map(lambda x: target[x], output_keys)
            invalid_values = filter(lambda x: not math.isfinite(x), output_values)
            invalid_count = sum(map(lambda x: 1, invalid_values))
            return invalid_count == 0

        return filter(is_output_valid, output_rows)

    def _parse_accumulator_key(self, key):
        """Deserialize an accumulator key indicating which accumulator should take in sample.

        Args:
            key: The serialized string key.

        Returns:
            Deserailized version of the accumulator key as a primitives-only dictionary.
        """
        pieces = key.split('\t')
        return {
            'geohash': pieces[0],
            'month': int(pieces[1]),
            'attr': pieces[2]
        }

    def _get_accumulator_key(self, geohash, month, attr):
        """Create a string identifying a specific accumulator which should recieve sample.

        Args:
            geohash: The name of the geohash for which sample is available.
            month: The month for which sample is available.
            attr: The name of the attribute like "chirps" for which sample is available.

        Returns:
            String unique to the accumulator matching the parameters.
        """
        pieces = [geohash, month, attr]
        pieces_str = map(lambda x: str(x), pieces)
        return '\t'.join(pieces_str)


def is_record_in_scope(target, threshold, geohash_sim_size=4, historic=False):
    """Determine if a record is in scope to be exported as part of a series.

    Args:
        target: The record as a primitives only dictionary to be evaluated.
        threshold: Loss threshold of the series being exported.
        geohash_sim_size: Geohash length of the series being exported.
        historic: Flag indicating if the series is historic or future data. True if historic and
            false if future.

    Returns:
        True if this record is in scope and false otherwise.
    """
    if threshold is not None and abs(float(target['threshold']) - threshold) > 0.0001:
        return False

    if abs(float(target['stdMult']) - 1) > 0.0001:
        return False

    if int(target['geohashSimSize']) != geohash_sim_size:
        return False

    if historic:
        if target['series'] != 'historic':
            return False
    else:
        if target['series'] == 'historic':
            return False

    if abs(float(target['unitSize'])) <= 1.0001:
        return False

    if USE_UNIT_FOR_COUNTERFACTUAL:
        return target['offsetBaseline'] == 'always'
    else:
        if '_counterfactual' in target['series']:
            return target['offsetBaseline'] == 'never'
        else:
            return target['offsetBaseline'] == 'always'


class SweepExportTask(luigi.Task):
    """Task for exporting information about the model training sweep."""

    def requires(self):
        """Require that the model sweep be completed.

        Returns:
            Dependencies for this task.
        """
        return {
            'normal': training_tasks.SweepTask(),
            'extended': training_tasks.SweepExtendedTask()
        }

    def output(self):
        """Indicate location where the sweep summary csv should be written.

        Returns:
            LocalTarget at which the summary CSV should be written.
        """
        return luigi.LocalTarget(const.get_file_location('export_sweep.csv'))

    def run(self):
        """Export the sweep information as a CSV file."""
        with self.output().open('w') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=SWEEP_OUTPUT_COLS)
            writer.writeheader()

            with self.input()['normal'].open('r') as f_in:
                reader = csv.DictReader(f_in)
                transformed_rows = map(lambda x: self._transform_row(x), reader)
                writer.writerows(transformed_rows)

            with self.input()['extended'].open('r') as f_in:
                reader = csv.DictReader(f_in)
                transformed_rows = map(lambda x: self._transform_row(x), reader)
                writer.writerows(transformed_rows)

    def _transform_row(self, target):
        """Standardize an output row.

        Args:
            target: The output row prior to standardization.

        Returns:
            Output row after standardization.
        """
        return {
            'block': target['block'],
            'layers': int(target['layers']),
            'l2Reg': float(target['l2Reg']),
            'dropout': float(target['dropout']),
            'allowCount': 1 if target['allowCount'].lower() == 'true' else 0,
            'trainMean': float(target['trainMean']),
            'trainStd': float(target['trainStd']),
            'validMean': float(target['validMean']),
            'validStd': float(target['validStd']),
            'testMean': float(target['testMean']),
            'testStd': float(target['testStd'])
        }


class GetFamilySizeTask(luigi.Task):
    """Get the number of groups (each with a statistical test) considered.

    Get the number of groups (each with a statistical test) considered as required by operations
    like the Bonferonni correction.
    """

    def requires(self):
        """Require that the simulations have been completed already to determine groups.

        Returns:
            CombineSimulationsTask
        """
        return sim_tasks.CombineSimulationsTask()

    def output(self):
        """Indicate where the family sizes should be written as a CSV.

        Returns:
            LocalTarget where the results are to be written.
        """
        return luigi.LocalTarget(const.get_file_location('export_count.json'))

    def run(self):
        """Find the family sizes and write to CSV."""
        with self.input().open('r') as f:
            reader = csv.DictReader(f)
            included_rows = filter(lambda x: is_record_in_scope(x, 0.25), reader)
            tuples = map(lambda x: (x['series'], 1), included_rows)
            reduced = toolz.itertoolz.reduceby(
                lambda x: x[0],
                lambda a, b: (a[0], a[1] + b[1]),
                tuples
            ).values()
            reduced_dict = dict(reduced)

        with self.output().open('w') as f:
            json.dump(reduced_dict, f)


class HistExportSubTask(luigi.Task):
    """Task which summarizes system-wide changes to yield within a simulation series.

    Task which summarizes system-wide changes to yield within a simulation series such as historic
    or 2050_SSP245. This will organize the data into a series of bins like a histogram but also
    provide summary statistics (claimsMpci, claimsSco, mean, count). Uses Luigi parameters:

     - geohash_size: The size of the geohash to summarize.
     - historic: True if should be historic data / baseline and false if future projections.
    """

    geohash_size = luigi.Parameter()
    historic = luigi.Parameter()

    def requires(self):
        """Require that the simulations have already been completed.

        Returns:
            CombineSimulationsTask
        """
        return sim_tasks.CombineSimulationsTask()

    def output(self):
        """Indicate where the bucketed system-wide simulation summary should be written.

        Returns:
            LocalTarget where the CSV is to be written.
        """
        vals = (self.geohash_size, 'historic' if self.historic else 'simulated')
        return luigi.LocalTarget(const.get_file_location('export_hist_%d_%s.csv' % vals))

    def run(self):
        """Summarize the simulations in a histogram-like structure."""
        with self.output().open('w') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=HIST_OUTPUT_COLS)
            writer.writeheader()

            with self.input().open() as f_in:
                all_rows = csv.DictReader(f_in)
                rows_allowed = filter(lambda x: self._get_is_target(x, 0.25), all_rows)
                rows_simplified = map(lambda x: self._simplify_input(x), rows_allowed)
                rows_combined = toolz.itertoolz.reduceby(
                    lambda x: x['series'],
                    lambda a, b: self._combine_inputs(a, b),
                    rows_simplified
                ).values()
                rows_interpreted_nested = map(lambda x: self._create_output_rows(x), rows_combined)
                rows_interpreted = itertools.chain(*rows_interpreted_nested)
                rows_interpreted_with_rename = map(
                    lambda x: self._rename_claims(x, 'claimsMpci'),
                    rows_interpreted
                )
                rows_with_meta = map(lambda x: self._add_meta(x), rows_interpreted_with_rename)
                writer.writerows(rows_with_meta)

            with self.input().open() as f_in:
                all_rows = csv.DictReader(f_in)
                rows_allowed = filter(lambda x: self._get_is_target(x, 0.15), all_rows)
                rows_simplified = map(lambda x: self._simplify_input(x), rows_allowed)
                rows_combined = toolz.itertoolz.reduceby(
                    lambda x: x['series'],
                    lambda a, b: self._combine_inputs(a, b),
                    rows_simplified
                ).values()
                rows_interpreted_nested = map(lambda x: self._create_output_rows(x), rows_combined)
                rows_interpreted = itertools.chain(*rows_interpreted_nested)
                rows_interpreted_with_rename = map(
                    lambda x: self._rename_claims(x, 'claimsSco'),
                    rows_interpreted
                )
                rows_claims = filter(
                    lambda x: x['bin'] == 'claimsSco',
                    rows_interpreted_with_rename
                )
                rows_with_meta = map(lambda x: self._add_meta(x), rows_claims)
                writer.writerows(rows_with_meta)

    def _simplify_input(self, target):
        """Parse a raw input record from the simulations task, interpreting floats.

        Args:
            target: The row to simplify.

        Returns:
            The input row after parsing.
        """
        num = float(target['num'])
        claims_rate = float(target['predictedClaims'])
        ret_dict = {
            'series': target['series'],
            'num': num,
            'predictedChange': float(target['predictedChange']),
            'claims': num * claims_rate
        }

        bin_keys = map(lambda x: 'bin%d' % x, sim_tasks.BINS)

        for key in bin_keys:
            ret_dict[key] = float(target[key])

        return ret_dict

    def _combine_inputs(self, a, b):
        """Combine two different input records by pooling their outputs statistically.

        Args:
            a: The first record to pool.
            b: The second record to pool.

        Returns:
            Results after pooling.
        """
        assert a['series'] == b['series']

        def get_weighted_avg(a_val, a_weight, b_val, b_weight):
            return (a_val * a_weight + b_val * b_weight) / (a_weight + b_weight)

        ret_dict = {
            'series': a['series'],
            'num': a['num'] + b['num'],
            'claims': a['claims'] + b['claims'],
            'predictedChange': get_weighted_avg(
                a['predictedChange'],
                a['num'],
                b['predictedChange'],
                b['num']
            )
        }

        bin_keys = map(lambda x: 'bin%d' % x, sim_tasks.BINS)

        for key in bin_keys:
            ret_dict[key] = a[key] + b[key]

        return ret_dict

    def _create_output_rows(self, target):
        """Standardize the format of output rows being written to the output CSV.

        Args:
            target: The record to standardize.

        Returns:
            The record after standardization.
        """
        is_counterfactual = '_counterfactual' in target['series']

        if target['series'] == 'historic':
            year = 2010
        else:
            year = int(target['series'].split('_')[0])

        def make_record_for_bin(bin_val):
            key = 'bin%d' % bin_val
            value = target[key]
            return {
                'set': year,
                'series': 'counterfactual' if is_counterfactual else 'predicted',
                'bin': bin_val,
                'val': value
            }

        bin_rows = map(make_record_for_bin, sim_tasks.BINS)

        def get_sum_below(threshold):
            bins = filter(lambda x: x <= threshold, sim_tasks.BINS)
            bin_keys = map(lambda x: 'bin%d' % x, bins)
            return sum(map(lambda x: target[x], bin_keys))

        claims = target['claims']
        total_count = target['num']
        mean_val = target['predictedChange']

        def make_summary_row(key, value):
            return {
                'set': year,
                'series': 'counterfactual' if is_counterfactual else 'predicted',
                'bin': key,
                'val': value
            }

        summary_rows = [
            make_summary_row('claims', claims),
            make_summary_row('mean', mean_val),
            make_summary_row('cnt', total_count)
        ]

        return itertools.chain(bin_rows, summary_rows)

    def _rename_claims(self, target, new_name):
        """Rename the claims column to another value.

        Args:
            target: The record in which "claims" should be renamed.
            new_name: The new name to give to "claims" which may be more descriptive like
                "claimsMpci" or similar.

        Returns:
            The input record after renaming claims.
        """
        if target['bin'] == 'claims':
            return {
                'set': target['set'],
                'series': target['series'],
                'bin': new_name,
                'val': target['val']
            }
        else:
            return target

    def _add_meta(self, target):
        """Add information about the simulation like geohash size to an input record.

        Args:
            target: The record to which additional information should be added.

        Returns:
            The input record after adding metadata like geohashSize.
        """
        return {
            'geohashSize': int(self.geohash_size),
            'set': target['set'],
            'series': target['series'],
            'bin': target['bin'],
            'val': target['val']
        }

    def _get_is_target(self, candidate, threshold):
        """Determine if the given record is in scope and reports on the given loss threshold.

        Args:
            candidate: The record to check.
            threshold: The loss threshold to check.

        Returns:
            True if the record is in scope and reports on the given threshold. False otherwise.
        """
        return is_record_in_scope(
            candidate,
            threshold,
            geohash_sim_size=self.geohash_size,
            historic=self.historic
        )


class HistExportTask(luigi.Task):
    """Export a combination of all histogram exports."""

    def requires(self):
        """List of histogram tasks to be concatenated.

        Returns:
            List of histograms to concatenate.
        """
        return [
            HistExportSubTask(geohash_size=4, historic=False),
            HistExportSubTask(geohash_size=4, historic=True),
            HistExportSubTask(geohash_size=5, historic=False),
            HistExportSubTask(geohash_size=5, historic=True)
        ]

    def output(self):
        """Indicate location where to write the concatenated output.

        Returns:
            LocalTarget where the output should be written.
        """
        return luigi.LocalTarget(const.get_file_location('export_hist.csv'))

    def run(self):
        """Concatenate histograms data."""
        with self.output().open('w') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=HIST_OUTPUT_COLS)
            writer.writeheader()

            for target in self.input():
                with target.open('r') as f_in:
                    reader = csv.DictReader(f_in)
                    writer.writerows(reader)


class SummaryExportTemplateTask(luigi.Task):
    """Abstract base class template to export a summary of a simulation.

    Abstract base class serving as a template for tasks which export the tool summary file used to
    describe geohash-level results.
    """

    def requires(self):
        """Get data pipeline tasks to be summarized.

        Returns:
            Tasks whose outputs are to be summarized.
        """
        return {
            'sim': sim_tasks.CombineSimulationsTask(),
            'familySize': GetFamilySizeTask()
        }

    def output(self):
        """Indicate location where to write the concatenated output.

        Returns:
            LocalTarget where the output should be written.
        """
        return luigi.LocalTarget(const.get_file_location(self.get_filename()))

    def run(self):
        """Create and write the summary out to disk."""
        with self.input()['familySize'].open() as f:
            family_sizes = json.load(f)

        with self.input()['sim'].open() as f:
            records = csv.DictReader(f)
            records_allowed = filter(lambda x: self._get_is_record_allowed(x), records)
            inputs = map(lambda x: self._simplify_input(x, family_sizes), records_allowed)
            reduced = toolz.itertoolz.reduceby(
                lambda x: self._get_input_key(x),
                lambda a, b: self._combine_input(a, b),
                inputs
            ).values()

        reduced_with_location = map(lambda x: self._finalize_record(x), reduced)

        with self.output().open('w') as f:
            writer = csv.DictWriter(f, fieldnames=TOOL_OUTPUT_COLS)
            writer.writeheader()
            writer.writerows(reduced_with_location)

    def get_filename(self):
        """Get the name of the file to be written in the workspace directory.

        Returns:
            String filename but not file path.
        """
        raise NotImplementedError('Use implementor.')

    def get_geohash_size(self):
        """Get the size of the geohash used in the simulation being summarized.

        Returns:
            Geohash length like 4.
        """
        raise NotImplementedError('Use implementor.')

    def _simplify_input(self, target, family_sizes):
        """Parse an input row from the simulation raw results.

        Args:
            target: The row to be parsed.
            family_sizes: Dictionary mapping name of series to number of statistical tests run in
                that series.

        Returns:
            The input row parsed, simplified, and standardized.
        """
        is_counterfactual = '_counterfactual' in target['series']
        year = int(target['year'])

        if target['series'] == 'historic':
            year_series_str = 'historic'
        else:
            year_series = int(target['series'].split('_')[0])
            year_series_str = '%d_SSP245' % year_series

        def include_if_predicted(value):
            if is_counterfactual:
                return None
            else:
                return value

        def include_if_counterfactual(value):
            if is_counterfactual:
                return value
            else:
                return None

        series = target['series']
        family_size = family_sizes[series]
        predicted_change = float(target['predictedChange'])
        predicted_claims = float(target['predictedClaims'])
        p = float(target['p'])

        return {
            'geohash': target['geohash'],
            'isCounterfactual': is_counterfactual,
            'year': year,
            'condition': year_series_str,
            'lossThreshold': float(target['threshold']),
            'num': float(target['num']),
            'predictedRisk': include_if_predicted(predicted_claims),
            'counterfactualRisk': include_if_counterfactual(predicted_claims),
            'p': include_if_predicted(p),
            'pMeets0.05n': include_if_predicted(p < 0.05 / family_size),
            'pMeets0.10n': include_if_predicted(p < 0.10 / family_size),
            'predictedMean': include_if_predicted(predicted_change),
            'counterfactualMean': include_if_counterfactual(predicted_change),
            'original': target
        }

    def _get_input_key(self, target):
        """Get a key uniquely identifying a geohash in a simulation.

        Args:
            target: The record for which a key is being requested.

        Returns:
            Key identifying a geohash in a simulation as a string.
        """
        pieces = [target['geohash'], target['year'], target['condition'], target['lossThreshold']]
        pieces_str = map(lambda x: str(x), pieces)
        return '\t'.join(pieces_str)

    def _combine_input(self, a, b):
        """Combine two input rows.

        Args:
            a: The first sample to combine.
            b: The second sample to combine.

        Returns:
            Combined input record after pooling those samples.
        """
        a_is_counterfactual = a['isCounterfactual']
        b_is_counterfactual = b['isCounterfactual']
        invalid_flag = a_is_counterfactual is None or b_is_counterfactual is None
        flags_match = a_is_counterfactual == b_is_counterfactual
        if invalid_flag or flags_match:
            print(a)
            print(b)
            raise RuntimeError('Merging incompatible records.')

        if a['isCounterfactual']:
            predicted_record = b
            counterfactual_record = a
        else:
            predicted_record = a
            counterfactual_record = b

        return {
            'geohash': predicted_record['geohash'],
            'isCounterfactual': None,
            'year': predicted_record['year'],
            'condition': predicted_record['condition'],
            'lossThreshold': predicted_record['lossThreshold'],
            'num': predicted_record['num'],
            'predictedRisk': predicted_record['predictedRisk'],
            'counterfactualRisk': counterfactual_record['counterfactualRisk'],
            'p': predicted_record['p'],
            'pMeets0.05n': predicted_record['pMeets0.05n'],
            'pMeets0.10n': predicted_record['pMeets0.10n'],
            'predictedMean': predicted_record['predictedMean'],
            'counterfactualMean': counterfactual_record['counterfactualMean']
        }

    def _finalize_record(self, target):
        """Add metadata and supporting information to an output record.

        Args:
            target: The record to extend.

        Returns:
            Finalized record.
        """
        geohash = target['geohash']
        geohash_decoded = geolib.geohash.decode(geohash)

        return {
            'geohash': geohash,
            'year': target['year'],
            'condition': target['condition'],
            'lossThreshold': target['lossThreshold'],
            'num': target['num'],
            'predictedRisk': target['predictedRisk'],
            'counterfactualRisk': target['counterfactualRisk'],
            'p': target['p'],
            'pMeets0.05n': 1 if target['pMeets0.05n'] else 0,
            'pMeets0.10n': 1 if target['pMeets0.10n'] else 0,
            'predictedMean': target['predictedMean'],
            'counterfactualMean': target['counterfactualMean'],
            'lat': geohash_decoded.lat,
            'lng': geohash_decoded.lon
        }

    def _get_is_record_allowed(self, target):
        """Determie if an input record is in scope and should be included.

        Args:
            target: The record to check.

        Returns:
            True if in scope and false otherwise.
        """
        return is_record_in_scope(target, None, geohash_sim_size=self.get_geohash_size())


class SummaryExportTask(SummaryExportTemplateTask):
    """Task which exports the 4 character geohash export task."""

    def get_filename(self):
        """Get the filename where the output should be written.

        Returns:
            String filename where the summary should be written.
        """
        return 'export_summary.csv'

    def get_geohash_size(self):
        """Get the geohash size for which the summary should be generated.

        Returns:
            Geohash length like 4.
        """
        return 4


class SummaryExportLongTask(SummaryExportTemplateTask):
    """Task which exports the 5 character geohash export task."""

    def get_filename(self):
        """Get the filename where the output should be written.

        Returns:
            String filename where the summary should be written.
        """
        return 'export_summary_5char.csv'

    def get_geohash_size(self):
        """Get the geohash size for which the summary should be generated.

        Returns:
            Geohash length like 4.
        """
        return 5


class CombinedTasksRecordTask(luigi.Task):
    """Get a record of all simulation tasks."""

    def requires(self):
        """Get the individual series task generation tasks.

        Returns:
            Mapping from name of series to task generation task.
        """
        return {
            'historic': sim_tasks.MakeSimulationTasksHistoricTask(),
            '2010 series': sim_tasks.MakeSimulationTasks2010Task(),
            '2030 series': sim_tasks.MakeSimulationTasks2030Task(),
            '2050 series': sim_tasks.MakeSimulationTasks2050Task()
        }

    def output(self):
        """Get the location at which the task CSV should be written.

        Returns:
            LocalTarget at which the output should be written.
        """
        return luigi.LocalTarget(const.get_file_location('export_combined_tasks.csv'))

    def run(self):
        """Build the output summary."""
        with self.output().open('w') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=COMBINED_TASK_FIELDS)
            writer.writeheader()

            for series in self.input().keys():
                with self.input()[series].open('r') as f_in:
                    reader = csv.DictReader(f_in)
                    writer.writerows(reader)


class ExportClaimsRatesTemplateTask(luigi.Task):
    """Determine the rate at which claims are expected in different simluations."""

    def requires(self):
        """Require that the simulations be complete.

        Returns:
            CombineSimulationsTask
        """
        raise NotImplementedError('Use implementor.')

    def output(self):
        """Get the location where information about the claims rate should be written.

        Returns:
            LocalTarget at which the claims rate should be written.
        """
        raise NotImplementedError('Use implementor.')

    def run(self):
        """Calculate claims rate."""
        with self.output().open('w') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=CLAIMS_COLS)
            writer.writeheader()

            with self.input().open('r') as f_in:
                reader = csv.DictReader(f_in)
                simplified = map(lambda x: self._simplify_input(x), reader)
                reduced = toolz.itertoolz.reduceby(
                    lambda x: self._key_record(x),
                    lambda a, b: self._combine_records(a, b),
                    simplified
                ).values()

                writer.writerows(reduced)

    def _simplify_input(self, target):
        """Parse, standardize, and simplify a raw simulation output record.

        Args:
            target: The record to simplify.

        Returns:
            The record after simplification and standardization.
        """
        def determine_year(target):
            if target <= 2016:
                return 2010
            elif target <= 2039:
                return 2030
            else:
                return 2050
        return {
            'offsetBaseline': target['offsetBaseline'],
            'year': determine_year(int(target['year'])),
            'condition': target['condition'],
            'threshold': float(target['threshold']),
            'thresholdStd': float(target['thresholdStd']),
            'stdMult': float(target['stdMult']),
            'geohashSimSize': float(target['geohashSimSize']),
            'num': float(target['num']),
            'claimsRate': float(target['predictedClaims']),
            'claimsRateStd': float(target['predictedClaimsStd'])
        }

    def _combine_records(self, a, b):
        """Combine two different sampels from the outpu simulation.

        Args:
            a: The first sample to pool.
            b: The second sample to pool.

        Returns:
            The combined sample.
        """
        def get_weighted(key):
            a_val = a[key]
            a_num = a['num']
            b_val = b[key]
            b_num = b['num']

            return (a_val * a_num + b_val * b_num) / (a_num + b_num)

        return {
            'offsetBaseline': a['offsetBaseline'],
            'year': a['year'],
            'condition': a['condition'],
            'threshold': a['threshold'],
            'thresholdStd': a['thresholdStd'],
            'stdMult': a['stdMult'],
            'geohashSimSize': a['geohashSimSize'],
            'num': a['num'] + b['num'],
            'claimsRate': get_weighted('claimsRate'),
            'claimsRateStd': get_weighted('claimsRateStd')
        }

    def _key_record(self, record):
        """Get the claims rate group key for a record.

        Get the unique key describing a group (simulation, condition, threshold, etc) in which a
        claims rate may be calculated.

        Args:
            record: The record for which a group key should be found.

        Returns:
            String group key.
        """
        def get_value(key):
            if key in CLAIMS_RATE_GROUP_KEYS_FLOAT:
                return '%.4f' % record[key]
            else:
                return record[key]

        key_vals = map(get_value, CLAIMS_RATE_GROUP_KEYS)
        key_vals_str = map(lambda x: str(x), key_vals)
        return '\t'.join(key_vals_str)


class ExportClaimsRatesTask(ExportClaimsRatesTemplateTask):
    """Determine the rate at which claims are expected in different simluations."""

    def requires(self):
        """Require that the simulations be complete.

        Returns:
            CombineSimulationsTask
        """
        return sim_tasks.CombineSimulationsTask()

    def output(self):
        """Get the location where information about the claims rate should be written.

        Returns:
            LocalTarget at which the claims rate should be written.
        """
        return luigi.LocalTarget(const.get_file_location('export_claims.csv'))


class ExportClaimsRatesHoldYearTask(ExportClaimsRatesTemplateTask):
    """Determine rate at which claims are expected in different simluations without year change."""

    def requires(self):
        """Require that the simulations be complete.

        Returns:
            CombineSimulationsTask
        """
        return sim_tasks.CombineSimulationsHoldYearTask()

    def output(self):
        """Get the location where information about the claims rate should be written.

        Returns:
            LocalTarget at which the claims rate should be written.
        """
        return luigi.LocalTarget(const.get_file_location('export_claims_hold_year.csv'))
