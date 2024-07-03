import concurrent.futures
import csv
import itertools
import json
import random
import sqlite3
import statistics

import coiled
import keras
import luigi
import pandas
import toolz

import cluster_tasks
import const
import normalize_tasks
import selection_tasks
import training_tasks

BINS = list(map(lambda x: x - 100, range(0, 205, 5)))
BIN_FIELDS = ['bin%d' % x for x in BINS]
OUTPUT_FIELDS = [
    'offsetBaseline',
    'unitSize',
    'geohash',
    'year',
    'condition',
    'threshold',
    'stdMult',
    'geohashSimSize',
    'num',
    'predictedChange',
    'baselineChange',
    'adaptedChange',
    'predictedClaims',
    'baselineClaims',
    'adaptedClaims',
    'predictedLoss',
    'baselineLoss',
    'adaptedLoss',
    'p',
    'pAdapted'
] + BIN_FIELDS
NUM_ARGS = 4
STD_MULT = [1.0]
THRESHOLDS = [0.25, 0.15]
GEOHASH_SIZE = [4, 5]
SAMPLE_MODEL_RESIDUALS = True


class Task:

    def __init__(self, geohash, year, condition, original_mean, original_std, projected_mean,
        projected_std, num_observations):
        self._geohash = geohash
        self._year = year
        self._condition = condition
        self._original_mean = original_mean
        self._original_std = original_std
        self._projected_mean = projected_mean
        self._projected_std = projected_std
        self._num_observations = num_observations
        
    def get_geohash(self):
        return self._geohash
    
    def get_year(self):
        return self._year
    
    def get_condition(self):
        return self._condition
    
    def get_original_mean(self):
        return self._original_mean
    
    def get_original_std(self):
        return self._original_std
    
    def get_projected_mean(self):
        return self._projected_mean
    
    def get_projected_std(self):
        return self._projected_std
    
    def get_num_observations(self):
        return self._num_observations


def run_simulation(task, deltas, threshold, std_mult, geohash_sim_size, offset_baseline, unit_size):
    import math
    import random

    import distribution_struct

    import scipy.stats
    import toolz.itertoolz
    
    mean_deltas = deltas['mean']
    std_deltas = deltas['std']

    original_mean = task.get_original_mean()
    original_std = task.get_original_std()
    projected_mean = task.get_projected_mean()
    projected_std = task.get_projected_std()
    num_observations = task.get_num_observations() / const.RESOLUTION_SCALER

    if geohash_sim_size == 5:
        num_observations = round(num_observations / 32)

    if num_observations == 0:
        return None

    unit_size_scaled = math.ceil(unit_size / const.RESOLUTION_SCALER)
    num_units = round(num_observations / unit_size_scaled)

    if num_units == 0:
        return None
    
    predicted_deltas = []
    baseline_deltas = []
    adapted_deltas = []
    for unit_i in range(0, num_units):
        predicted_yield_acc = distribution_struct.WelfordAccumulator()
        adapted_yield_acc = distribution_struct.WelfordAccumulator()

        for pixel_i in range(0, unit_size_scaled):

            if SAMPLE_MODEL_RESIDUALS:
                mean_delta = random.choice(mean_deltas) * -1
                std_delta = random.choice(std_deltas) * -1
            else:
                mean_delta = 0
                std_delta = 0

            sim_mean = projected_mean + mean_delta
            sim_std = projected_std * std_mult + std_delta
            
            predicted_yield = random.gauss(mu=sim_mean, sigma=sim_std)
            adapted_yield = random.gauss(mu=sim_mean + sim_std, sigma=sim_std)

            predicted_yield_acc.add(predicted_yield)
            adapted_yield_acc.add(adapted_yield)

        def execute_offset_baseline(value, offset):
            # The average will partially catch up during the period
            half_way = (value + offset) / 2
            effective_offset = (half_way + offset * 3) / 4
            return value - effective_offset

        if offset_baseline == 'always':
            predicted_delta = execute_offset_baseline(
                predicted_yield_acc.get_mean(),
                original_mean
            )
            adapted_delta = execute_offset_baseline(
                adapted_yield_acc.get_mean(),
                original_mean
            )
        elif offset_baseline == 'negative' and original_mean < 0:
            predicted_delta = execute_offset_baseline(
                predicted_yield_acc.get_mean(),
                original_mean
            )
            adapted_delta = execute_offset_baseline(
                adapted_yield_acc.get_mean(),
                original_mean
            )
        else:
            predicted_delta = predicted_yield_acc.get_mean()
            adapted_delta = adapted_yield_acc.get_mean()
        
        baseline_deltas.append(original_mean)
        predicted_deltas.append(predicted_delta)
        adapted_deltas.append(adapted_delta)

    def get_claims_rate(target):
        neg_threshold = threshold * -1
        claims = filter(lambda x: x <= neg_threshold, target)
        num_claims = sum(map(lambda x: 1, claims))
        return num_claims / len(target)
    
    def get_loss_level(target):
        neg_threshold = threshold * -1
        claims = list(filter(lambda x: x <= neg_threshold, target))
        if len(claims) > 0:
            return statistics.mean(claims)
        else:
            return 0

    def get_change(target):
        if len(target) > 0:
            return statistics.mean(target)
        else:
            return 0
    
    baseline_change = get_change(baseline_deltas)
    predicted_change = get_change(predicted_deltas)
    adapted_change = get_change(adapted_deltas)

    baseline_claims_rate = get_claims_rate(baseline_deltas)
    predicted_claims_rate = get_claims_rate(predicted_deltas)
    adapted_claims_rate = get_claims_rate(adapted_deltas)
    
    baseline_loss = get_loss_level(baseline_deltas)
    predicted_loss = get_loss_level(predicted_deltas)
    adapted_loss = get_loss_level(adapted_deltas)

    p_baseline = scipy.stats.mannwhitneyu(predicted_deltas, baseline_deltas)[1]
    p_adapted = scipy.stats.mannwhitneyu(predicted_deltas, adapted_deltas)[1]

    ret_dict = {
        'offsetBaseline': offset_baseline,
        'unitSize': unit_size,
        'geohash': task.get_geohash(),
        'year': task.get_year(),
        'condition': task.get_condition(),
        'threshold': threshold,
        'stdMult': std_mult,
        'geohashSimSize': geohash_sim_size,
        'num': num_observations,
        'predictedChange': predicted_change,
        'baselineChange': baseline_change,
        'adaptedChange': adapted_change,
        'predictedClaims': predicted_claims_rate,
        'baselineClaims': baseline_claims_rate,
        'adaptedClaims': adapted_claims_rate,
        'predictedLoss': predicted_loss,
        'baselineLoss': baseline_loss,
        'adaptedLoss': adapted_loss,
        'p': p_baseline,
        'pAdapted': p_adapted
    }

    def cap_delta(target):
        if target > 1:
            return 1
        elif target < -1:
            return -1
        else:
            return target

    def project_delta(target):
        percent = target * 100
        percent_rounded = round(percent / 5) * 5
        return percent_rounded

    predicted_deltas_cap = map(cap_delta, predicted_deltas)
    predicted_deltas_projected = map(project_delta, predicted_deltas_cap)
    deltas_tuples = map(lambda x: (x, 1), predicted_deltas_projected)
    deltas_count = dict(toolz.itertoolz.reduceby(
        lambda x: x[0],
        lambda a, b: (a[0], a[1] + b[1]),
        deltas_tuples
    ).values())

    for bin_num in BINS:
        ret_dict['bin%d' % bin_num] = deltas_count.get(bin_num, 0)

    return ret_dict


def parse_record(record_raw):
    geohash = str(record_raw[0])
    year = int(record_raw[1])
    condition = str(record_raw[2])
    original_mean = float(record_raw[3])
    original_std = float(record_raw[4])
    projected_mean = float(record_raw[5])
    projected_std = float(record_raw[6])
    num_observations = int(record_raw[7])
    
    return Task(
        geohash,
        year,
        condition,
        original_mean,
        original_std,
        projected_mean,
        projected_std,
        num_observations
    )


def parse_record_dict(record_raw):
    geohash = str(record_raw['geohash'])
    year = int(record_raw['year'])
    condition = str(record_raw['condition'])
    original_mean = float(record_raw['originalYieldMean'])
    original_std = float(record_raw['originalYieldStd'])
    projected_mean = float(record_raw['projectedYieldMean'])
    projected_std = float(record_raw['projectedYieldStd'])
    num_observations = int(record_raw['numObservations'])
    
    return Task(
        geohash,
        year,
        condition,
        original_mean,
        original_std,
        projected_mean,
        projected_std,
        num_observations
    )


def run_simulation_set(tasks, deltas, threshold, std_mult, geohash_sim_size, offset_baseline, unit_size):
    results_all = map(
        lambda x: run_simulation(x, deltas, threshold, std_mult, geohash_sim_size, offset_baseline, unit_size),
        tasks
    )
    results_valid = filter(lambda x: x is not None, results_all)
    results_realized = list(results_valid)
    return results_realized


class NormalizeRefHistoricTrainingFrameTask(luigi.Task):

    def requires(self):
        return normalize_tasks.NormalizeHistoricTrainingFrameTask()

    def output(self):
        return luigi.LocalTarget(const.get_file_location('ref_historic_normalized.csv'))

    def run(self):
        with self.input().open('r') as f_in:
            reader = csv.DictReader(f_in)
            allowed_rows = filter(lambda x: int(x['year']) in const.FUTURE_REF_YEARS, reader)

            with self.output().open('w') as f_out:
                writer = csv.DictWriter(f_out, fieldnames=const.TRAINING_FRAME_ATTRS)
                writer.writeheader()
                writer.writerows(allowed_rows)


class GetNumObservationsTask(luigi.Task):

    def requires(self):
        return NormalizeRefHistoricTrainingFrameTask()

    def output(self):
        return luigi.LocalTarget(const.get_file_location('observation_counts.csv'))

    def run(self):
        with self.input().open('r') as f_in:
            with self.output().open('w') as f_out:
                rows = csv.DictReader(f_in)
                standard_rows = map(lambda x: self._standardize_row(x), rows)

                writer = csv.DictWriter(f_out, fieldnames=['geohash', 'year', 'yieldObservations'])
                writer.writeheader()
                writer.writerows(standard_rows)

    def _standardize_row(self, target):
        return {
            'geohash': target['geohash'],
            'year': int(target['year']),
            'yieldObservations': int(target['yieldObservations'])
        }


class ProjectTaskTemplate(luigi.Task):

    def requires(self):
        return {
            'model': selection_tasks.TrainFullModel(),
            'target': self.get_target_task(),
            'configuration': selection_tasks.SelectConfigurationTask()
        }

    def output(self):
        return luigi.LocalTarget(const.get_file_location(self.get_filename()))

    def run(self):
        target_frame = pandas.read_csv(self.input()['target'].path)
        
        with self.input()['configuration'].open('r') as f:
            configuration = json.load(f)['constrained']

        model = keras.models.load_model(self.input()['model'].path)
        
        additional_block = configuration['block']
        allow_count = configuration['allowCount'].lower() == 'true'
        
        input_attrs = training_tasks.get_input_attrs(additional_block, allow_count)
        inputs = target_frame[input_attrs]

        target_frame['joinYear'] = target_frame['year']
        target_frame['simYear'] = target_frame['year'] - 2007 + self.get_base_year()
        target_frame['year'] = target_frame['simYear']

        outputs = model.predict(inputs)
        target_frame['predictedMean'] = outputs[:,0]
        target_frame['predictedStd'] = outputs[:,1]

        target_frame[[
            'geohash',
            'simYear',
            'joinYear',
            'predictedMean',
            'predictedStd',
            'yieldObservations'
        ]].to_csv(self.output().path)

    def get_target_task(self):
        raise NotImplementedError('Use implementor.')
    
    def get_base_year(self):
        raise NotImplementedError('Use implementor.')

    def get_filename(self):
        raise NotImplementedError('Use implementor.')


class InterpretProjectTaskTemplate(luigi.Task):

    def requires(self):
        return {
            'target': self.get_target_task(),
            'dist': normalize_tasks.GetInputDistributionsTask()
        }

    def output(self):
        return luigi.LocalTarget(const.get_file_location(self.get_filename()))

    def run(self):
        with self.input()['dist'].open('r') as f:
            rows = csv.DictReader(f)

            distributions = {}

            for row in rows:
                distributions[row['field']] = {
                    'mean': float(row['mean']),
                    'std': float(row['std'])
                }

        with self.output().open('w') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=[
                'geohash',
                'simYear',
                'joinYear',
                'predictedMean',
                'predictedStd',
                'yieldObservations'
            ])
            writer.writeheader()
            
            with self.input()['target'].open('r') as f_in:
                reader = csv.DictReader(f_in)
                standardized_rows = map(lambda x: self._standardize_row(x), reader)
                updated_rows = map(lambda x: self._update_row(x, distributions), standardized_rows)
                writer.writerows(updated_rows)

    def get_target_task(self):
        raise NotImplementedError('Must use implementor.')

    def get_filename(self):
        raise NotImplementedError('Must use implementor.')

    def _standardize_row(self, target):
        return {
            'geohash': target['geohash'],
            'simYear': int(target['simYear']),
            'joinYear': int(target['joinYear']),
            'predictedMean': float(target['predictedMean']),
            'predictedStd': float(target['predictedStd']),
            'yieldObservations': int(target['yieldObservations'])
        }

    def _update_row(self, row, distributions):
        mean_dist = distributions['yieldMean']
        std_dist = distributions['yieldStd']

        original_predicted_mean = row['predictedMean']
        original_predicted_std = row['predictedStd']

        def interpret(target, dist):
            if const.NORM_YIELD_FIELDS:
                reverse_z = target * dist['std'] + dist['mean']
                return reverse_z
            else:
                return target

        interpreted_predicted_mean = interpret(original_predicted_mean, mean_dist)
        interpreted_predicted_std = interpret(original_predicted_std, std_dist)

        row['predictedMean'] = interpreted_predicted_mean
        row['predictedStd'] = interpreted_predicted_std

        return row


class MakeSimulationTasksTemplate(luigi.Task):

    def requires(self):
        return {
            'baseline': self.get_baseline_task(),
            'projection': self.get_projection_task(),
            'numObservations': GetNumObservationsTask()
        }

    def output(self):
        return luigi.LocalTarget(const.get_file_location(self.get_filename()))

    def run(self):
        baseline_indexed = self._index_input('baseline')
        projection_indexed = self._index_input('projection')

        baseline_keys = set(baseline_indexed.keys())
        projection_keys = set(projection_indexed)
        keys = baseline_keys.intersection(projection_keys)

        with self.input()['numObservations'].open('r') as f:
            rows = csv.DictReader(f)
            keyed = map(
                lambda x: ('%s.%s' % (x['geohash'], x['year']), int(x['yieldObservations'])),
                rows
            )
            observation_counts_indexed = dict(keyed)

        def get_num_observations(geohash, join_year):
            key = '%s.%d' % (geohash, join_year)
            return observation_counts_indexed.get(key, 0)

        def get_output_row(key):
            baseline_record = baseline_indexed[key]
            projection_record = projection_indexed[key]

            geohash = projection_record['geohash']
            join_year = projection_record['joinYear']
            
            return {
                'geohash': geohash,
                'year': projection_record['simYear'],
                'condition': self.get_condition(),
                'originalYieldMean': baseline_record['predictedMean'],
                'originalYieldStd': baseline_record['predictedStd'],
                'projectedYieldMean': projection_record['predictedMean'],
                'projectedYieldStd': projection_record['predictedStd'],
                'numObservations': get_num_observations(geohash, join_year)
            }

        output_rows_all = map(get_output_row, keys)
        output_rows = filter(lambda x: x['numObservations'] > 0, output_rows_all)

        with self.output().open('w') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'geohash',
                'year',
                'condition',
                'originalYieldMean',
                'originalYieldStd',
                'projectedYieldMean',
                'projectedYieldStd',
                'numObservations'
            ])
            writer.writeheader()
            writer.writerows(output_rows)

    def get_filename(self):
        raise NotImplementedError('Use implementor.')
    
    def get_baseline_task(self):
        raise NotImplementedError('Use implementor.')
    
    def get_projection_task(self):
        raise NotImplementedError('Use implementor.')

    def get_condition(self):
        raise NotImplementedError('Use implementor.')

    def _index_input(self, name):
        indexed = {}
        
        with self.input()[name].open('r') as f:
            rows_raw = csv.DictReader(f)
            rows = map(lambda x: self._parse_row(x), rows_raw)
            
            for row in rows:
                key = '%s.%d' % (row['geohash'], row['joinYear'])
                indexed[key] = row

        return indexed

    def _parse_row(self, row):
        return {
            'geohash': row['geohash'],
            'simYear': int(row['simYear']),
            'joinYear': int(row['joinYear']),
            'predictedMean': float(row['predictedMean']),
            'predictedStd': float(row['predictedStd']),
            'yieldObservations': int(row['yieldObservations'])
        }


class ExecuteSimulationTasksTemplate(luigi.Task):

    def requires(self):
        return {
            'tasks': self.get_tasks_task(),
            'deltas': self.get_deltas_task(),
            'cluster': cluster_tasks.StartClusterTask()
        }

    def output(self):
        return luigi.LocalTarget(const.get_file_location(self.get_filename()))

    def run(self):
        with self.input()['tasks'].open('r') as f:
            rows = csv.DictReader(f)
            tasks = [parse_record_dict(x) for x in rows]

        job_shuffles = list(range(0, 200))
        input_records_grouped = toolz.itertoolz.groupby(
            lambda x: random.choice(job_shuffles),
            tasks
        )

        tasks_with_variations = list(
            itertools.product(
                input_records_grouped.values(),
                THRESHOLDS,
                STD_MULT,
                GEOHASH_SIZE,
                ['always', 'never'],
                [const.UNIT_SIZE_IN_PIXELS, 1]
            )
        )

        cluster = cluster_tasks.get_cluster()
        cluster.adapt(minimum=20, maximum=100)
        client = cluster.get_client()

        with self.input()['deltas'].open('r') as f:
            rows = csv.DictReader(f)
            test_rows = filter(lambda x: x['setAssign'] == 'test', rows)
            rows_mean_std_linear_str = map(
                lambda x: (x['meanResidual'], x['stdResidual']),
                test_rows
            )
            rows_mean_std_linear = map(
                lambda x: (float(x[0]), float(x[1])),
                rows_mean_std_linear_str
            )
            unzipped = list(zip(*rows_mean_std_linear))
            deltas = {
                'mean': unzipped[0],
                'std': unzipped[1]
            }

        outputs_all = client.map(
            lambda x: run_simulation_set(x[0], deltas, x[1], x[2], x[3], x[4], x[5]),
            tasks_with_variations
        )
        outputs_realized = map(lambda x: x.result(), outputs_all)

        with self.output().open('w') as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()

            for output_set in outputs_realized:
                writer.writerows(output_set)
                f.flush()

    def get_tasks_task(self):
        raise NotImplementedError('Use implementor.')

    def get_filename(self):
        raise NotImplementedError('Use implementor.')

    def get_deltas_task(self):
        raise NotImplementedError('Use implementor.')


class ProjectHistoricTask(ProjectTaskTemplate):
    
    def get_target_task(self):
        return NormalizeRefHistoricTrainingFrameTask()
    
    def get_base_year(self):
        return 2007

    def get_filename(self):
        return 'historic_project_dist.csv'


class Project2030Task(ProjectTaskTemplate):
    
    def get_target_task(self):
        return normalize_tasks.NormalizeFutureTrainingFrameTask(condition='2030_SSP245')
    
    def get_base_year(self):
        return 2030

    def get_filename(self):
        return '2030_project_dist.csv'


class Project2030CounterfactualTask(ProjectTaskTemplate):
    
    def get_target_task(self):
        return NormalizeRefHistoricTrainingFrameTask()
    
    def get_base_year(self):
        return 2030

    def get_filename(self):
        return '2030_project_dist_counterfactual.csv'


class Project2050Task(ProjectTaskTemplate):
    
    def get_target_task(self):
        return normalize_tasks.NormalizeFutureTrainingFrameTask(condition='2050_SSP245')
    
    def get_base_year(self):
        return 2050

    def get_filename(self):
        return '2050_project_dist.csv'


class Project2050CounterfactualTask(ProjectTaskTemplate):
    
    def get_target_task(self):
        return NormalizeRefHistoricTrainingFrameTask()
    
    def get_base_year(self):
        return 2050

    def get_filename(self):
        return '2050_project_dist_counterfactual.csv'


class InterpretProjectHistoricTask(InterpretProjectTaskTemplate):
    
    def get_target_task(self):
        return ProjectHistoricTask()

    def get_filename(self):
        return 'historic_project_dist_interpret.csv'


class InterpretProject2030Task(InterpretProjectTaskTemplate):
    
    def get_target_task(self):
        return Project2030Task()

    def get_filename(self):
        return '2030_project_dist_interpret.csv'


class InterpretProject2030CounterfactualTask(InterpretProjectTaskTemplate):
    
    def get_target_task(self):
        return Project2030CounterfactualTask()

    def get_filename(self):
        return '2030_project_dist_counterfactual_interpret.csv'


class InterpretProject2050Task(InterpretProjectTaskTemplate):
    
    def get_target_task(self):
        return Project2050Task()

    def get_filename(self):
        return '2050_project_dist_interpret.csv'


class InterpretProject2050CounterfactualTask(InterpretProjectTaskTemplate):
    
    def get_target_task(self):
        return Project2050CounterfactualTask()

    def get_filename(self):
        return '2050_project_dist_counterfactual_interpret.csv'


class MakeSimulationTasksHistoricTask(MakeSimulationTasksTemplate):

    def get_filename(self):
        return 'current_sim_tasks.csv'
    
    def get_baseline_task(self):
        return InterpretProjectHistoricTask()
    
    def get_projection_task(self):
        return InterpretProjectHistoricTask()

    def get_condition(self):
        return 'historic'


class MakeSimulationTasks2030Task(MakeSimulationTasksTemplate):

    def get_filename(self):
        return '2030_sim_tasks.csv'
    
    def get_baseline_task(self):
        return InterpretProjectHistoricTask()
    
    def get_projection_task(self):
        return InterpretProject2030Task()

    def get_condition(self):
        return '2030_SSP245'


class MakeSimulationTasks2050Task(MakeSimulationTasksTemplate):

    def get_filename(self):
        return '2050_sim_tasks.csv'
    
    def get_baseline_task(self):
        return InterpretProject2030Task()
    
    def get_projection_task(self):
        return InterpretProject2050Task()

    def get_condition(self):
        return '2050_SSP245'


class MakeSimulationTasks2030CounterfactualTask(MakeSimulationTasksTemplate):

    def get_filename(self):
        return '2030_sim_tasks_counterfactual.csv'
    
    def get_baseline_task(self):
        return InterpretProjectHistoricTask()
    
    def get_projection_task(self):
        return InterpretProject2030CounterfactualTask()

    def get_condition(self):
        return '2030_SSP245'


class MakeSimulationTasks2050CounterfactualTask(MakeSimulationTasksTemplate):

    def get_filename(self):
        return '2050_sim_tasks_counterfactual.csv'
    
    def get_baseline_task(self):
        return InterpretProject2030CounterfactualTask()
    
    def get_projection_task(self):
        return InterpretProject2050CounterfactualTask()

    def get_condition(self):
        return '2030_SSP245'


class ExecuteSimulationTasksHistoricPredictedTask(ExecuteSimulationTasksTemplate):

    def get_tasks_task(self):
        return MakeSimulationTasksHistoricTask()

    def get_filename(self):
        return 'historic_sim.csv'

    def get_deltas_task(self):
        return selection_tasks.PostHocTestRawDataTemporalTask()


class ExecuteSimulationTasks2030PredictedTask(ExecuteSimulationTasksTemplate):

    def get_tasks_task(self):
        return MakeSimulationTasks2030Task()

    def get_filename(self):
        return '2030_sim.csv'

    def get_deltas_task(self):
        return selection_tasks.PostHocTestRawDataTemporalTask()


class ExecuteSimulationTasks2050PredictedTask(ExecuteSimulationTasksTemplate):

    def get_tasks_task(self):
        return MakeSimulationTasks2050Task()

    def get_filename(self):
        return '2050_sim.csv'

    def get_deltas_task(self):
        return selection_tasks.PostHocTestRawDataTemporalTask()


class ExecuteSimulationTasks2030Counterfactual(ExecuteSimulationTasksTemplate):

    def get_tasks_task(self):
        return MakeSimulationTasks2030CounterfactualTask()

    def get_filename(self):
        return '2030_sim_counterfactual.csv'

    def get_deltas_task(self):
        return selection_tasks.PostHocTestRawDataTemporalTask()

class ExecuteSimulationTasks2050Counterfactual(ExecuteSimulationTasksTemplate):

    def get_tasks_task(self):
        return MakeSimulationTasks2050CounterfactualTask()

    def get_filename(self):
        return '2050_sim_counterfactual.csv'

    def get_deltas_task(self):
        return selection_tasks.PostHocTestRawDataTemporalTask()


class CombineSimulationsTasks(luigi.Task):

    def requires(self):
        return {
            'historic': ExecuteSimulationTasksHistoricPredictedTask(),
            '2030': ExecuteSimulationTasks2030PredictedTask(),
            '2030_counterfactual': ExecuteSimulationTasks2030Counterfactual(),
            '2050': ExecuteSimulationTasks2050PredictedTask(),
            '2050_counterfactual': ExecuteSimulationTasks2050Counterfactual()
        }

    def output(self):
        return luigi.LocalTarget(const.get_file_location('sim_combined.csv'))

    def run(self):
        with self.output().open('w') as f:
            writer = csv.DictWriter(f, fieldnames=['series'] + OUTPUT_FIELDS)
            writer.writeheader()
            self._write_out('historic', writer)
            self._write_out('2030', writer)
            self._write_out('2030_counterfactual', writer)
            self._write_out('2050', writer)
            self._write_out('2050_counterfactual', writer)

    def _write_out(self, label, writer):
        with self.input()[label].open('r') as f:
            reader = csv.DictReader(f)
            rows = map(lambda x: self._add_series(label, x), reader)
            writer.writerows(rows)

    def _add_series(self, series, row):
        row['series'] = series
        return row


class MakeSingleYearStatistics(luigi.Task):

    def requires(self):
        return CombineSimulationsTasks()

    def output(self):
        return luigi.LocalTarget(const.get_file_location('sim_combined_summary_1_year.csv'))

    def run(self):
        with self.input().open('r') as f:
            reader = csv.DictReader(f)

            def is_equal(value, target):
                value_float = float(value)
                return abs(value_float - target) < 0.00001

            right_threshold = filter(lambda x: is_equal(x['threshold'], 0.25), reader)
            right_std = filter(lambda x: is_equal(x['stdMult'], 1), right_threshold)
            right_geohash = filter(lambda x: is_equal(x['geohashSimSize'], 4), right_std)
            rows = map(lambda x: self._parse_row(x), right_geohash)
            rows_by_series = toolz.itertoolz.groupby('series', rows)

        def make_weight_record(trial):
            num = trial['num']
            return {
                'predictedLossWeightAcc': (1 - trial['predicted']) * num
            }

        def process_family(trials):
            num_trials = len(trials)
            threshold = 0.05 / num_trials
            significant = filter(lambda x: x['pAdapted'] < threshold, trials)

    def _parse_row(self, row):
        return {
            'series': row['series'],
            'num': int(row['num']),
            'predicted': float(row['predicted']),
            'p': float(row['p']),
            'pAdapted': float(row['pAdapted'])
        }
