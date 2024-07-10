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

HIST_OUTPUT_COLS = ['set', 'series', 'bin', 'val']

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

USE_UNIT_FOR_COUNTERFACTUAL = True


class ClimateExportTask(luigi.Task):

    def requires(self):
        return {
            'historic': normalize_tasks.NormalizeHistoricTrainingFrameTask(),
            '2030': normalize_tasks.NormalizeFutureTrainingFrameTask(condition='2030_SSP245'),
            '2050': normalize_tasks.NormalizeFutureTrainingFrameTask(condition='2050_SSP245')
        }

    def output(self):
        return luigi.LocalTarget(const.get_file_location('export_climate.csv'))

    def run(self):
        historic_accumulators = {}
        with self.input()['historic'].open() as f:
            reader = csv.DictReader(f)

            for row in reader:
                geohash = row['geohash']
                
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
        pieces = key.split('\t')
        return {
            'geohash': pieces[0],
            'month': int(pieces[1]),
            'attr': pieces[2]
        }

    def _get_accumulator_key(self, geohash, month, attr):
        pieces = [geohash, month, attr]
        pieces_str = map(lambda x: str(x), pieces)
        return '\t'.join(pieces_str)


def is_default_config_record(target, threshold):
    if threshold is not None and abs(float(target['threshold']) - threshold) > 0.0001:
        return False

    if abs(float(target['stdMult']) - 1) > 0.0001:
        return False

    if int(target['geohashSimSize']) != 4:
        return False

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

    def requires(self):
        return training_tasks.SweepTask()

    def output(self):
        return luigi.LocalTarget(const.get_file_location('export_sweep.csv'))

    def run(self):
        with self.input().open('r') as f_in:
            reader = csv.DictReader(f_in)
            transformed_rows = map(lambda x: self._transform_row(x), reader)

            with self.output().open('w') as f_out:
                writer = csv.DictWriter(f_out, fieldnames=SWEEP_OUTPUT_COLS)
                writer.writeheader()
                writer.writerows(transformed_rows)

    def _transform_row(self, target):
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

    def requires(self):
        return sim_tasks.CombineSimulationsTasks()

    def output(self):
        return luigi.LocalTarget(const.get_file_location('export_count.json'))

    def run(self):
        with self.input().open('r') as f:
            reader = csv.DictReader(f)
            included_rows = filter(lambda x: is_default_config_record(x, 0.25), reader)
            tuples = map(lambda x: (x['series'], 1), included_rows)
            reduced = toolz.itertoolz.reduceby(
                lambda x: x[0],
                lambda a, b: (a[0], a[1] + b[1]),
                tuples
            ).values()
            reduced_dict = dict(reduced)

        with self.output().open('w') as f:
            json.dump(reduced_dict, f)


class HistExportTask(luigi.Task):

    def requires(self):
        return sim_tasks.CombineSimulationsTasks()

    def output(self):
        return luigi.LocalTarget(const.get_file_location('export_hist.csv'))

    def run(self):
        with self.output().open('w') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=HIST_OUTPUT_COLS)
            writer.writeheader()

            with self.input().open() as f_in:
                all_rows = csv.DictReader(f_in)
                rows_allowed = filter(lambda x: is_default_config_record(x, 0.25), all_rows)
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
                writer.writerows(rows_interpreted_with_rename)

            with self.input().open() as f_in:
                all_rows = csv.DictReader(f_in)
                rows_allowed = filter(lambda x: is_default_config_record(x, 0.25), all_rows)
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
                writer.writerows(rows_claims)


    def _simplify_input(self, target):
        num = float(target['num'])
        claims_rate = float(target['predictedClaims'])
        ret_dict = {
            'series': target['series'],
            'num': num,
            'predictedChange': float(target['predictedChange']),
            'claims': num * claims_rate
        }

        bin_keys = map(lambda x: 'bin%d' % x, sim_tasks.BINS)

        for key in bin_keys:f
            ret_dict[key] = float(target[key])

        return ret_dict

    def _combine_inputs(self, a, b):
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
        is_counterfactual = '_counterfactual' in target['series']
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
        if target['bin'] == 'claims':
            return {
                'set': target['set'],
                'series': target['series'],
                'bin': new_name,
                'val': target['val']
            }
        else:
            return target


class SummaryExportTask(luigi.Task):

    def requires(self):
        return {
            'sim': sim_tasks.CombineSimulationsTasks(),
            'familySize': GetFamilySizeTask()
        }

    def output(self):
        return luigi.LocalTarget(const.get_file_location('export_summary.csv'))

    def run(self):
        with self.input()['familySize'].open() as f:
            family_sizes = json.load(f)

        with self.input()['sim'].open() as f:
            records = csv.DictReader(f)
            records_allowed = filter(lambda x: is_default_config_record(x, None), records)
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

    def _simplify_input(self, target, family_sizes):
        is_counterfactual = '_counterfactual' in target['series']
        year = int(target['year'])
        year_series = int(target['series'].split('_')[0])

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
            'condition': '%d_SSP245' % year_series,
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
        pieces = [target['geohash'], target['year'], target['condition'], target['lossThreshold']]
        pieces_str = map(lambda x: str(x), pieces)
        return '\t'.join(pieces_str)

    def _combine_input(self, a, b):
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


class CombinedTasksRecordTask(luigi.Task):

    def requires(self):
        return {
            'historic': sim_tasks.MakeSimulationTasksHistoricTask(),
            '2030 series': sim_tasks.MakeSimulationTasks2030Task(),
            '2050 series': sim_tasks.MakeSimulationTasks2050Task()
        }

    def output(self):
        return luigi.LocalTarget(const.get_file_location('export_combined_tasks.csv'))

    def run(self):
        with self.output().open('w') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=COMBINED_TASK_FIELDS)
            writer.writeheader()

            for series in self.input().keys():
                with self.input()[series].open('r') as f_in:
                    reader = csv.DictReader(f_in)
                    writer.writerows(reader)
