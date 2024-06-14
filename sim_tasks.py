import concurrent.futures
import csv
import itertools
import json
import random
import sqlite3
import statistics

import coiled
import toolz

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