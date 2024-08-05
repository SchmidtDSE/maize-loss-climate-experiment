import csv
import itertools
import os
import random
import shutil

import boto3
import luigi

import cluster_tasks
import const
import normalize_tasks

DEFAULT_NUM_LAYERS = [1, 2, 3, 4, 5, 6]
DEFAULT_REGULARIZATION = [0.00, 0.05, 0.10, 0.15, 0.20]
DEFAULT_DROPOUT = [0.00, 0.01, 0.05, 0.10, 0.50]
BLOCKS = [
    'all attrs'
]
BLOCKS_EXTENDED = [
    'year',
    'rhn',
    'rhx',
    'tmax',
    'tmin',
    'chirps',
    'svp',
    'vpd',
    'wbgtmax'
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
    'testStd'
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
        build_layer(512),
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

    model.compile(optimizer='adamw', loss='mae', metrics=['mae'])

    return model


def try_model(access_key, secret_key, num_layers, l2_reg, dropout, bucket_name, filename,
    additional_block, allow_count, seed=12345, output_attrs=OUTPUT_ATTRS, epochs=30,
    blocked_attrs=BLOCKED_ATTRS):
    import os
    import random

    import boto3
    import pandas

    import const
    import file_util

    random.seed(seed)

    using_local = access_key == '' or secret_key == ''

    if using_local:
        temp_file_path = os.path.join(bucket_name, filename)
    else:
        temp_file_path = '/tmp/' + filename

    input_attrs = get_input_attrs(additional_block, allow_count)

    def get_data():

        if not os.path.isfile(temp_file_path):
            if using_local:
                raise RuntimeError('Could not find ' + temp_file_path)
            else:
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
                'outputs': train[output_attrs],
                'weights': train[const.SAMPLE_WEIGHT_ATTR]
            },
            'valid': {
                'inputs': valid[input_attrs],
                'outputs': valid[output_attrs],
                'weights': valid[const.SAMPLE_WEIGHT_ATTR]
            },
            'test': {
                'inputs': test[input_attrs],
                'outputs': test[output_attrs],
                'weights': test[const.SAMPLE_WEIGHT_ATTR]
            }
        }

    def train_model(model, data_splits):
        model.fit(
            data_splits['train']['inputs'],
            data_splits['train']['outputs'],
            epochs=epochs,
            verbose=None,
            sample_weight=data_splits['train']['weights']
        )

    def evaluate_model(model, data_splits):
        def get_abs_diff(a, b):
            return abs(a - b)

        def get_maes(target_inputs, target_outputs, weights):
            predictions = model.predict(target_inputs, verbose=None)
            paired_flat = list(zip(predictions, target_outputs.to_numpy(), weights))

            paired_parsed = map(lambda x: {
                'mean': get_abs_diff(x[0][0], x[1][0]),
                'std': get_abs_diff(x[0][1], x[1][1]),
                'weight': x[2]
            }, paired_flat)
            paired_parsed_realized = list(paired_parsed)

            weight_sum = sum(map(lambda x: x['weight'], paired_parsed_realized))

            mean_mean_errors = sum(map(
                lambda x: x['mean'] * x['weight'],
                paired_parsed_realized
            )) / weight_sum

            mean_std_errors = sum(map(
                lambda x: x['std'] * x['weight'],
                paired_parsed_realized
            )) / weight_sum

            return {
                'mean': mean_mean_errors,
                'std': mean_std_errors
            }

        train_errors = get_maes(
            data_splits['train']['inputs'],
            data_splits['train']['outputs'],
            data_splits['train']['weights']
        )

        valid_errors = get_maes(
            data_splits['valid']['inputs'],
            data_splits['valid']['outputs'],
            data_splits['valid']['weights']
        )

        test_errors = get_maes(
            data_splits['test']['inputs'],
            data_splits['test']['outputs'],
            data_splits['test']['weights']
        )

        return {
            'block': additional_block,
            'layers': num_layers,
            'l2Reg': l2_reg,
            'dropout': dropout,
            'allowCount': allow_count,
            'trainMean': train_errors['mean'],
            'trainStd': train_errors['std'],
            'validMean': valid_errors['mean'],
            'validStd': valid_errors['std'],
            'testMean': test_errors['mean'],
            'testStd': test_errors['std']
        }

    if os.path.isfile(temp_file_path):
        file_util.remove_temp_file(
            temp_file_path,
            access_key,
            secret_key
        )

    split_data = get_data()
    model = build_model(num_layers, len(input_attrs), l2_reg, dropout)
    train_model(model, split_data)
    return evaluate_model(model, split_data)


class UploadHistoricTrainingFrame(luigi.Task):

    def requires(self):
        return normalize_tasks.NormalizeHistoricTrainingFrameTask()

    def output(self):
        return luigi.LocalTarget(const.get_file_location('upload_historic_confirm.txt'))

    def run(self):
        access_key = os.environ.get('AWS_ACCESS_KEY', '')
        access_secret = os.environ.get('AWS_ACCESS_SECRET', '')

        using_local = access_key == '' or access_secret == ''
        if using_local:
            input_path = self.input().path
            output_path = os.path.join(const.BUCKET_OR_DIR, const.HISTORIC_TRAINING_FILENAME)
            shutil.copy2(input_path, output_path)
        else:
            s3 = boto3.client(
                's3',
                aws_access_key_id=access_key,
                aws_secret_access_key=access_secret
            )

            s3.upload_file(self.input().path, const.BUCKET_OR_DIR, const.HISTORIC_TRAINING_FILENAME)

        with self.output().open('w') as f:
            f.write('success')


class SweepTemplateTask(luigi.Task):

    def requires(self):
        return {
            'upload': UploadHistoricTrainingFrame(),
            'cluster': cluster_tasks.StartClusterTask()
        }

    def output(self):
        return luigi.LocalTarget(const.get_file_location(self.get_filename()))

    def run(self):
        num_layers = self.get_num_layers()
        l2_regs = self.get_l2_regs()
        dropouts = self.get_dropouts()

        combinations = itertools.product(
            num_layers,
            l2_regs,
            dropouts,
            self.get_blocks(),
            self.get_allow_counts()
        )

        combinations_realized = list(combinations)
        random.shuffle(combinations_realized)

        access_key = os.environ.get('AWS_ACCESS_KEY', '')
        access_secret = os.environ.get('AWS_ACCESS_SECRET', '')

        cluster = cluster_tasks.get_cluster()
        cluster.adapt(minimum=10, maximum=self.get_max_workers())

        client = cluster.get_client()
        outputs = client.map(
            lambda x: try_model(
                access_key,
                access_secret,
                x[0],
                x[1],
                x[2],
                const.BUCKET_OR_DIR,
                const.HISTORIC_TRAINING_FILENAME,
                x[3],
                x[4]
            ),
            combinations_realized
        )

        with self.output().open('w') as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            for output in map(lambda x: x.result(), outputs):
                writer.writerow(output)
                f.flush()

    def get_filename(self):
        raise NotImplementedError('Must use implementor.')

    def get_num_layers(self):
        raise NotImplementedError('Must use implementor.')

    def get_l2_regs(self):
        raise NotImplementedError('Must use implementor.')

    def get_dropouts(self):
        raise NotImplementedError('Must use implementor.')

    def get_blocks(self):
        raise NotImplementedError('Must use implementor.')

    def get_max_workers(self):
        raise NotImplementedError('Must use implementor.')

    def get_allow_counts(self):
        raise NotImplementedError('Must use implementor.')


class SweepTask(SweepTemplateTask):

    def get_filename(self):
        return 'sweep.csv'

    def get_num_layers(self):
        return DEFAULT_NUM_LAYERS

    def get_l2_regs(self):
        return DEFAULT_REGULARIZATION

    def get_dropouts(self):
        return DEFAULT_DROPOUT

    def get_blocks(self):
        return BLOCKS

    def get_max_workers(self):
        return 300

    def get_allow_counts(self):
        return [True, False]


class SweepExtendedTask(SweepTemplateTask):

    def get_filename(self):
        return 'sweep_extended.csv'

    def get_num_layers(self):
        return DEFAULT_NUM_LAYERS

    def get_l2_regs(self):
        return DEFAULT_REGULARIZATION

    def get_dropouts(self):
        return DEFAULT_DROPOUT

    def get_blocks(self):
        return BLOCKS_EXTENDED

    def get_max_workers(self):
        return 500

    def get_allow_counts(self):
        return [True]
