import csv
import itertools
import json
import os
import sys

import const
import normalize_tasks

DEFAULT_NUM_LAYERS = [1, 2, 3, 4, 5]
DEFAULT_REGULARIZATION = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
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
    'trainMean',
    'trainStd',
    'validMean',
    'validStd',
    'testMean',
    'testStd'
]


def try_model(access_key, secret_key, num_layers, l2_reg, dropout, bucket_name, filename,
    input_attrs, additional_block, seed=12345, output_attrs=OUTPUT_ATTRS, epochs=30,
    blocked_attrs=BLOCKED_ATTRS):
    import csv
    import os
    import random
    import statistics

    import boto3
    import keras
    import toolz.itertoolz

    import normalize_tasks

    random.seed(seed)

    input_attrs = list(filter(lambda x: additional_block not in x, input_attrs))
    temp_file_path = '/tmp/' + filename

    def get_data():

        if not os.path.isfile(temp_file_path):
            s3 = boto3.resource(
                's3',
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key
            )

            s3_object = s3.Object(bucket_name, filename)
            s3_object.download_file(temp_file_path)

        def get_year_set(year):
            if year < 2013:
                return 'train'
            else:
                return 'test' if year % 2 == 0 else 'valid'
        
        def assign_by_year(row):
            row['setAssign'] = get_year_set(row['year'])
            return row

        with open(temp_file_path, 'r') as f:
            records_raw = csv.DictReader(f)
            records_parsed = map(lambda x: normalize_tasks.parse_row(x), records_raw)
            records_assigned = map(lambda x: assign_by_year(x), records_parsed)
            records_grouped = toolz.itertoolz.groupby('setAssign', records_assigned)

        get_inputs = lambda (target): [target[x] for x in input_attrs]
        get_outputs = lambda (target): [target[x] for x in output_attrs]

        records_grouped_items = records_grouped.items()
        records_grouped_org = map(
            lambda x: (x[0], {'inputs': get_inputs(x[1]), 'outputs': get_outputs(x[1])}),
            records_grouped_items
        )
        return dict(records_grouped_org)

    def build_model(num_inputs):
        model = keras.Sequential()
        model.add(keras.Input(shape=(num_inputs,)))
        model.add(keras.layers.Normalization())
        
        layers = [
            keras.layers.Dense(256, activation='leaky_relu'),
            keras.layers.Dense(128, activation='leaky_relu'),
            keras.layers.Dense(64, activation='leaky_relu'),
            keras.layers.Dense(32, activation='leaky_relu'),
            keras.layers.Dense(8, activation='leaky_relu')
        ][-num_layers:]
        
        for layer in layers:
            model.add(layer)
            if dropout > 0:
                model.add(keras.layers.Dropout(dropout))
        
        model.add(keras.layers.Dense(2, activation='linear'))

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        return model

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

        def get_maes(target_inputs, target_outputs):
            predictions = model.predict(target_inputs, verbose=None)
            paired_flat = zip(predictions, target_outputs.to_numpy())
            paired_parsed = map(lambda x: {
                'mean': get_abs_diff(x[0][0], x[1][0]),
                'std': get_abs_diff(x[0][1], x[1][1])
            }, paired_flat)
            paired_parsed_realized = list(paired_parsed)

            mean_mean_errors = statistics.mean(map(lambda x: x['mean'], paired_parsed_realized))
            mean_std_errors = statistics.mean(map(lambda x: x['std'], paired_parsed_realized))

            median_mean_errors = statistics.median(map(lambda x: x['mean'], paired_parsed_realized))
            median_std_errors = statistics.median(map(lambda x: x['std'], paired_parsed_realized))
            
            return {
                'mean': {'mean': mean_mean_errors, 'median': median_mean_errors},
                'std': {'mean': mean_std_errors, 'median': median_std_errors}
            }

        train_errors = get_maes(data_splits['train']['inputs'], data_splits['train']['outputs'])
        valid_errors = get_maes(data_splits['valid']['inputs'], data_splits['valid']['outputs'])
        test_errors = get_maes(data_splits['test']['inputs'], data_splits['test']['outputs'])
        
        return {
            'block': additional_block,
            'layers': num_layers,
            'l2Reg': l2_reg,
            'dropout': dropout,
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
            'testStdMedian': test_errors['std']['median']
        }

    if os.path.isfile(temp_file_path):
        os.remove(temp_file_path)

    split_data = get_data()
    model = build_model(len(input_attrs))
    train_model(model, split_data)
    return evaluate_model(model, split_data)


class UploadHistoricTrainingFrame(luigi.Task):

    def requires(self):
        return normalize_tasks.NormalizeHistoricTrainingFrame()

    def output(self):
        return luigi.LocalTarget('upload_historic_confirm.txt')

    def run(self):



class SweepTask(luigi.Task):

    def requires(self):
        return UploadHistoricTrainingFrame()

    def output(self):
        return luigi.LocalTarget('sweep.csv')

    def run(self):
        num_layers = DEFAULT_NUM_LAYERS
        l2_regs = DEFAULT_REGULARIZATION
        dropouts = DEFAULT_DROPOUT

        all_attrs = const.TRAINING_FRAME_ATTRS
        all_attrs_no_geohash = filter(lambda x: x != 'geohash', all_attrs)
        all_attrs_no_output = filter(lambda x: x not in OUTPUT_ATTRS, all_attrs_no_geohash)
        all_attrs_no_count = filter(lambda x: 'count' not in x.lower(), all_attrs_no_output)
        input_attrs = sorted(filter(lambda x: x not in BLOCKED_ATTRS, all_attrs_no_count))
        combinations = itertools.product(num_layers, l2_regs, dropouts, BLOCKS)

        access_key = os.environ['CLIMATE_ACCESS_KEY']
        access_secret = os.environ['CLIMATE_ACCESS_SECRET']

        client = cluster_tasks.get_cluster()
        cluster.adapt(minimum=10, maximum=200)

        outputs = client.map(
            lambda x: try_model(
                access_key,
                access_secret,
                x[0],
                x[1],
                x[2],
                bucket,
                filename,
                input_attrs,
                x[3]
            ),
            list(combinations)
        )

        with self.output().open('w') as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            for output in map(lambda x: x.result(), outputs):
                writer.writerow(output)
                f.flush()
