import csv
import itertools
import json
import os
import sys

import boto3
import luigi

import cluster_tasks
import const
import normalize_tasks

DEFAULT_NUM_LAYERS = [1, 2, 3, 4, 5]
DEFAULT_REGULARIZATION = [0.00, 0.01, 0.02, 0.05, 0.07, 0.10, 0.20, 0.50, 0.70]
DEFAULT_DROPOUT = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
BLOCKS = [
    'all attrs',
    'year',
    'soil',
    'rhn',
    'rhx',
    'tmax',
    'tmin',
    'chirps',
    'svp',
    'vpt',
    'wbgt',
    'greendays'
]
BLOCKED_ATTRS = {
    'geohash',
    'geohashAgg',
    'yieldMean',
    'yieldStd',
    'yieldObservations',
    'setAssign'
}
OUTPUT_ATTRS = sorted(['yieldMean', 'yieldStd'])
OUTPUT_FIELDS = [
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
    'testStd',
    'trainMeanMedian',
    'trainStdMedian',
    'validMeanMedian',
    'validStdMedian',
    'testMeanMedian',
    'testStdMedian',
    'trainPercentMean',
    'trainPercentStd',
    'validPercentMean',
    'validPercentStd',
    'testPercentMean',
    'testPercentStd',
    'trainPercentMeanMedian',
    'trainPercentStdMedian',
    'validPercentMeanMedian',
    'validPercentStdMedian',
    'testPercentMeanMedian',
    'testPercentStdMedian'
]


def get_input_attrs(additional_block, allow_count):
    all_attrs = const.TRAINING_FRAME_ATTRS
    all_attrs_no_geohash = filter(lambda x: x != 'geohash', all_attrs)
    all_attrs_no_output = filter(lambda x: x not in OUTPUT_ATTRS, all_attrs_no_geohash)
    input_attrs = sorted(filter(lambda x: x not in BLOCKED_ATTRS, all_attrs_no_output))

    additional_block_lower = additional_block.lower()
    input_attrs = filter(lambda x: additional_block_lower not in x.lower(), input_attrs)

    if allow_count:
        input_attrs = filter(lambda x: 'count' not in x.lower(), input_attrs)

    input_attrs = list(input_attrs)

    return input_attrs


def build_model(num_layers, num_inputs, l2_reg, dropout):
    import keras

    model = keras.Sequential()
    model.add(keras.Input(shape=(num_inputs,)))
    model.add(keras.layers.Normalization())

    if l2_reg == 0:
        build_layer = lambda x: keras.layers.Dense(x, activation='leaky_relu')
    else:
        build_layer = lambda x: keras.layers.Dense(
            x,
            activation='leaky_relu',
            activity_regularizer=keras.regularizers.L2(l2_reg)
        )
    
    layers = [
        build_layer(256),
        build_layer(128),
        build_layer(64),
        build_layer(32),
        build_layer(8)
    ][-num_layers:]
    
    for layer in layers:
        model.add(layer)
        if dropout > 0:
            model.add(keras.layers.Dropout(dropout))
    
    model.add(keras.layers.Dense(2, activation='linear'))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model


def try_model(access_key, secret_key, num_layers, l2_reg, dropout, bucket_name, filename,
    additional_block, allow_count, seed=12345, output_attrs=OUTPUT_ATTRS, epochs=30,
    blocked_attrs=BLOCKED_ATTRS):
    import csv
    import os
    import random
    import statistics

    import boto3
    import pandas
    import toolz.itertoolz

    import normalize_tasks

    random.seed(seed)

    temp_file_path = '/tmp/' + filename

    input_attrs = get_input_attrs(additional_block, allow_count)

    def get_data():

        if not os.path.isfile(temp_file_path):
            s3 = boto3.resource(
                's3',
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key
            )

            s3_object = s3.Object(bucket_name, filename)
            s3_object.download_file(temp_file_path)

        def assign_year(year):
            if year < 2013:
                return 'train'
            else:
                return 'test' if year % 2 == 0 else 'valid'

        frame = pandas.read_csv(temp_file_path)        
        frame['setAssign'] = frame['year'].apply(assign_year)
        train = frame[frame['setAssign'] == 'train']
        valid = frame[frame['setAssign'] == 'valid']
        test = frame[frame['setAssign'] == 'test']

        return {
            'train': {
                'inputs': train[input_attrs],
                'outputs': train[output_attrs]
            },
            'valid': {
                'inputs': valid[input_attrs],
                'outputs': valid[output_attrs]
            },
            'test': {
                'inputs': test[input_attrs],
                'outputs': test[output_attrs]
            }
        }

    def train_model(model, data_splits):
        model.fit(
            data_splits['train']['inputs'],
            data_splits['train']['outputs'],
            epochs=epochs,
            verbose=None
        )

    def evaluate_model(model, data_splits):
        def get_abs_diff(a, b):
            return abs(a - b)

        def get_percent_diff(a, b):
            if b == 0 and a != 0:
                return 1
            else:
                return abs((a - b) / b)

        def get_maes(target_inputs, target_outputs):
            predictions = model.predict(target_inputs, verbose=None)
            paired_flat = list(zip(predictions, target_outputs.to_numpy()))
            
            paired_parsed = map(lambda x: {
                'mean': get_abs_diff(x[0][0], x[1][0]),
                'std': get_abs_diff(x[0][1], x[1][1]),
                'meanPercent': get_percent_diff(x[0][0], x[1][0]),
                'stdPercent': get_percent_diff(x[0][1], x[1][1])
            }, paired_flat)
            paired_parsed_realized = list(paired_parsed)

            mean_mean_errors = statistics.mean(map(lambda x: x['mean'], paired_parsed_realized))
            mean_std_errors = statistics.mean(map(lambda x: x['std'], paired_parsed_realized))

            median_mean_errors = statistics.median(map(lambda x: x['mean'], paired_parsed_realized))
            median_std_errors = statistics.median(map(lambda x: x['std'], paired_parsed_realized))

            mean_mean_errors_percent = statistics.mean(map(lambda x: x['meanPercent'], paired_parsed_realized))
            median_mean_errors_percent = statistics.mean(map(lambda x: x['stdPercent'], paired_parsed_realized))
            
            mean_std_errors_percent = statistics.median(map(lambda x: x['meanPercent'], paired_parsed_realized))
            median_std_errors_percent = statistics.median(map(lambda x: x['stdPercent'], paired_parsed_realized))
            
            return {
                'mean': {
                    'mean': mean_mean_errors,
                    'median': median_mean_errors,
                    'meanPercent': mean_mean_errors_percent,
                    'medianPercent': median_mean_errors_percent
                },
                'std': {
                    'mean': mean_std_errors,
                    'median': median_std_errors,
                    'meanPercent': mean_std_errors_percent,
                    'medianPercent': median_std_errors_percent
                }
            }

        train_errors = get_maes(data_splits['train']['inputs'], data_splits['train']['outputs'])
        valid_errors = get_maes(data_splits['valid']['inputs'], data_splits['valid']['outputs'])
        test_errors = get_maes(data_splits['test']['inputs'], data_splits['test']['outputs'])
        
        return {
            'block': additional_block,
            'layers': num_layers,
            'l2Reg': l2_reg,
            'dropout': dropout,
            'allowCount': allow_count,
            'trainMean': train_errors['mean']['mean'],
            'trainStd': train_errors['std']['mean'],
            'validMean': valid_errors['mean']['mean'],
            'validStd': valid_errors['std']['mean'],
            'testMean': test_errors['mean']['mean'],
            'testStd': test_errors['std']['mean'],
            'trainMeanMedian': train_errors['mean']['median'],
            'trainStdMedian': train_errors['std']['median'],
            'validMeanMedian': valid_errors['mean']['median'],
            'validStdMedian': valid_errors['std']['median'],
            'testMeanMedian': test_errors['mean']['median'],
            'testStdMedian': test_errors['std']['median'],
            'trainPercentMean': train_errors['mean']['meanPercent'],
            'trainPercentStd': train_errors['std']['meanPercent'],
            'validPercentMean': valid_errors['mean']['meanPercent'],
            'validPercentStd': valid_errors['std']['meanPercent'],
            'testPercentMean': test_errors['mean']['meanPercent'],
            'testPercentStd': test_errors['std']['meanPercent'],
            'trainPercentMeanMedian': train_errors['mean']['medianPercent'],
            'trainPercentStdMedian': train_errors['std']['medianPercent'],
            'validPercentMeanMedian': valid_errors['mean']['medianPercent'],
            'validPercentStdMedian': valid_errors['std']['medianPercent'],
            'testPercentMeanMedian': test_errors['mean']['medianPercent'],
            'testPercentStdMedian': test_errors['std']['medianPercent']
        }

    if os.path.isfile(temp_file_path):
        os.remove(temp_file_path)

    split_data = get_data()
    model = build_model(num_layers, len(input_attrs), l2_reg, dropout)
    train_model(model, split_data)
    return evaluate_model(model, split_data)


class UploadHistoricTrainingFrame(luigi.Task):

    def requires(self):
        return normalize_tasks.NormalizeHistoricTrainingFrame()

    def output(self):
        return luigi.LocalTarget(const.get_file_location('upload_historic_confirm.txt'))

    def run(self):
        access_key = os.environ['CLIMATE_ACCESS_KEY']
        access_secret = os.environ['CLIMATE_ACCESS_SECRET']

        s3 = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=access_secret
        )

        s3.upload_file(self.input().path, const.BUCKET_NAME, const.HISTORIC_TRAINING_FILENAME)

        with self.output().open('w') as f:
            f.write('success')


class SweepTask(luigi.Task):

    def requires(self):
        return {
            'upload': UploadHistoricTrainingFrame(),
            'cluster': cluster_tasks.StartClusterTask()
        }

    def output(self):
        return luigi.LocalTarget(const.get_file_location('sweep.csv'))

    def run(self):
        num_layers = DEFAULT_NUM_LAYERS
        l2_regs = DEFAULT_REGULARIZATION
        dropouts = DEFAULT_DROPOUT

        combinations = itertools.product(num_layers, l2_regs, dropouts, BLOCKS, [True, False])

        access_key = os.environ['CLIMATE_ACCESS_KEY']
        access_secret = os.environ['CLIMATE_ACCESS_SECRET']

        cluster = cluster_tasks.get_cluster()
        cluster.adapt(minimum=10, maximum=500)
        
        client = cluster.get_client()
        outputs = client.map(
            lambda x: try_model(
                access_key,
                access_secret,
                x[0],
                x[1],
                x[2],
                const.BUCKET_NAME,
                const.HISTORIC_TRAINING_FILENAME,
                x[3],
                x[4]
            ),
            list(combinations)
        )

        with self.output().open('w') as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            for output in map(lambda x: x.result(), outputs):
                writer.writerow(output)
                f.flush()
