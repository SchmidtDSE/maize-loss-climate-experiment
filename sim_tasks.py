import concurrent.futures
import csv
import itertools
import json
import random
import sqlite3
import statistics

import coiled
import luigi
import toolz

import selection_tasks
import training_tasks

OUTPUT_FIELDS = [
    'geohash',
    'year',
    'condition',
    'threshold',
    'stdMult',
    'geohashSimSize',
    'num',
    'predicted',
    'counterfactual',
    'adapted',
    'predictedLoss',
    'counterfactualLoss',
    'adaptedLoss',
    'pCounterfactual',
    'pAdapted'
]
NUM_ARGS = 4
STD_MULT = [0.5, 1.0, 1.5]
THRESHOLDS = [0.25, 0.15, 0.05]
GEOHASH_SIZE = [4, 5]


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


def run_simulation(task, deltas, threshold, std_mult, geohash_sim_size):
    import random

    import scipy.stats
    
    mean_deltas = deltas['mean']['deltas']
    std_deltas = deltas['std']['deltas']

    original_mean = task.get_original_mean()
    original_std = task.get_original_std()
    projected_mean = task.get_projected_mean()
    projected_std = task.get_projected_std()
    num_observations = task.get_num_observations()

    if geohash_sim_size == 5:
        num_observations = round(num_observations / 32)

    if num_observations == 0:
        num_observations = 1
    
    predicted_deltas = []
    counterfactual_deltas = []
    adapted_deltas = []
    for i in range(0, num_observations):
        mean_delta = random.choice(mean_deltas) * -1
        std_delta = random.choice(std_deltas) * -1
        sim_mean = projected_mean + mean_delta
        sim_std = projected_std * std_mult + std_delta
        
        prior_yield = random.gauss(mu=original_mean, sigma=original_std)
        predicted_yield = random.gauss(mu=sim_mean, sigma=sim_std)
        counterfactual_yield = random.gauss(mu=original_mean, sigma=original_std)
        adapted_yield = random.gauss(mu=sim_mean + sim_std, sigma=sim_std)

        predicted_delta = (predicted_yield - prior_yield) / prior_yield
        counterfactual_delta = (counterfactual_yield - prior_yield) / prior_yield
        adapted_delta = (adapted_yield - prior_yield) / prior_yield
        
        predicted_deltas.append(predicted_delta)
        counterfactual_deltas.append(counterfactual_delta)
        adapted_deltas.append(adapted_delta)

    def get_claims_rate(target):
        neg_threshold = threshold * -1
        claims = filter(lambda x: x <= neg_threshold, target)
        num_claims = sum(map(lambda x: 1, claims))
        return num_claims / len(target)
    
    def get_loss_level(target):
        neg_threshold = threshold * -1
        claims = filter(lambda x: x <= neg_threshold, target)
        return statistics.mean(claims)
    
    predicted_claims_rate = get_claims_rate(predicted_deltas)
    counterfactual_claims_rate = get_claims_rate(counterfactual_deltas)
    adapted_claims_rate = get_claims_rate(adapted_deltas)
    
    predicted_loss = get_loss_level(predicted_deltas)
    counterfactual_loss = get_loss_level(counterfactual_deltas)
    adapted_loss = get_loss_level(adapted_deltas)

    p_counterfactual = scipy.stats.mannwhitneyu(predicted_deltas, counterfactual_deltas)[1]
    p_adapted = scipy.stats.mannwhitneyu(predicted_deltas, adapted_deltas)[1]

    return {
        'geohash': task.get_geohash(),
        'year': task.get_year(),
        'condition': task.get_condition(),
        'threshold': threshold,
        'stdMult': std_mult,
        'geohashSimSize': geohash_sim_size,
        'num': num_observations,
        'predicted': predicted_claims_rate,
        'counterfactual': counterfactual_claims_rate,
        'adapted': adapted_claims_rate,
        'predictedLoss': predicted_loss,
        'counterfactualLoss': counterfactual_loss,
        'adaptedLoss': adapted_loss,
        'pCounterfactual': p_counterfactual,
        'pAdapted': p_adapted
    }


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


def run_simulation_set(tasks, deltas, threshold, std_mult, geohash_sim_size):
    return [run_simulation(x, deltas, threshold, std_mult, geohash_sim_size) for x in tasks]


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
        
        with self.input()['configuration'] as f:
            configuration = json.load(f)['constrained']

        model = keras.models.load_model(self.output().path)
        
        additional_block = configuration['block']
        allow_count = configuration['allowCount'].lower() == 'true'
        
        input_attrs = training_tasks.get_input_attrs(additional_block, allow_count)
        inputs = target_frame[input_attrs]

        target_frame['joinYear'] = target_frame['year']
        target_frame['simYear'] = target_frame['year'] - 2007 + 5 + self.get_base_year()

        target_frame['combinedOutput'] = model.predict(inputs)
        target_frame['predictedMean'] = target_frame['combinedOutput'].map(lambda x: x[0])
        target_frame['predictedStd'] = target_frame['combinedOutput'].map(lambda x: x[0])

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


class MakeSimulationTasksTemplate(luigi.Task):

    def requires(self):
        return {
            'baseline': self.get_baseline_task(),
            'projection': self.get_projection_task()
        }

    def output(self):
        return luigi.LocalTarget(const.get_file_location(self.get_filename()))

    def run(self):
        baseline_indexed = self._index_input('baseline')
        projection_indexed = self._index_input('projection')

        baseline_keys = set(baseline_indexed.keys())
        projection_keys = set(projection_indexed)
        keys = baseline_keys.intersection(projection_keys)

        def get_output_row(key):
            baseline_record = baseline_indexed[key]
            projection_record = projection_indexed[key]
            
            return {
                'geohash': projection_record['geohash'],
                'year': projection_record['simYear'],
                'condition': self.get_condition(),
                'originalYieldMean': baseline_record['projectedMean'],
                'originalYieldStd': baseline_record['projectedStd'],
                'projectedYieldMean': projection_record['projectedMean'],
                'projectedYieldStd': projection_record['projectedStd'],
                'numObservations': projection_record['yieldObservations']
            }

        output_rows = map(get_output_row, keys)

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
            'cluster': cluster_tasks.StartClusterTask()
        }

    def output(self):
        return luigi.LocalTarget(const.get_file_location(self.get_filename()))

    def run(self):
        with self.input().open('r') as f:
            rows = csv.DictReader(f)
            tasks = [parse_record_dict(x) for x in rows]

        job_shuffles = list(range(0, 500))
        input_records_grouped = toolz.itertoolz.groupby(
            lambda x: random.choice(job_shuffles),
            tasks
        )

        tasks_with_variations = list(
            itertools.product(input_records_grouped.values(), THRESHOLDS, STD_MULT, GEOHASH_SIZE)
        )

        cluster = cluster_tasks.get_cluster()
        cluster.adapt(minimum=10, maximum=500)
        client = cluster.get_client()

        outputs = client.map(
            lambda x: run_simulation_set(x[0], deltas, x[1], x[2], x[3]),
            tasks_with_variations
        )

        outputs_realized = map(lambda x: x.result(), outputs)

        with open(output_loc, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()

            for output_set in outputs_realized:
                writer.writerows(output_set)
                f.flush()

    def get_tasks_task(self):
        raise NotImplementedError('Use implementor.')
