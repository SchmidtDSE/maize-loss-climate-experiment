"""Tasks to train the neural network-based regressor.

License:
    BSD
"""
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
    'rhn',
    'rhx',
    'tmax',
    'tmin',
    'chirps',
    'svp',
    'vpd',
    'wbgtmax'
] + (['year'] if const.INCLUDE_YEAR_IN_MODEL else [])
BLOCKED_ATTRS = set([
    'year',
    'geohash',
    'geohashAgg',
    'yieldMean',
    'yieldStd',
    'yieldObservations',
    'setAssign'
] + ([] if const.INCLUDE_YEAR_IN_MODEL else ['year']))
OUTPUT_ATTRS = sorted(['yieldMean', 'yieldStd', 'yieldSkew', 'yieldKurtosis'])
OUTPUT_FIELDS = [
    'block',
    'layers',
    'l2Reg',
    'dropout',
    'allowCount',
    'trainMean',
    'trainStd',
    'trainSkew',
    'trainKurtosis',
    'validMean',
    'validStd',
    'validSkew',
    'validKurtosis',
    'testMean',
    'testStd',
    'testSkew',
    'testKurtosis'
]


def get_input_attrs(additional_block, allow_count):
    """Get the list of input attributes allowed to feed into the neural network.

    Args:
        additional_block: Collection of strings that are manually blocked from being used as neural
            network inputs.
        allow_count: Flag indicating if sample information can be included in model inputs. True if
            included and false otherwise.

    Returns:
        Sorted list of strings corresponding to attributes that can be used as neural network
        inputs.
    """
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


def build_model(num_layers, num_inputs, l2_reg, dropout, learning_rate=const.LEARNING_RATE):
    """Function to build a single model without fitting.

    Self-contained function to build a single model without fitting which can be exported to
    other machines for distributed computation.

    Args:
        num_layers: The number of internal hidden layers to use.
        num_inputs: The number of inputs the network should expect.
        l2_reg: Level of L2 regularization to apply to connections or 0 if no regularization.
        dropout: The dropout rate to apply for regularization or 0 if no dropout.

    Returns:
        Untrained keras model.
    """
    import keras
    import keras.optimizers

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

    model.add(keras.layers.Dense(4, activation='linear'))

    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])

    return model


def try_model(access_key, secret_key, num_layers, l2_reg, dropout, bucket_name, filename,
    additional_block, allow_count, seed=12345, output_attrs=OUTPUT_ATTRS, epochs=const.EPOCHS,
    blocked_attrs=BLOCKED_ATTRS):
    """Try building and training a model.

    Self-contained function to try building and training a model with imports such that it can be
    exported to other machines or used in distributed computing.

    Args:
        access_key: The AWS access key to use to get training data or empty string ('') if training
            data are local.
        secret_key: The AWS secret key to use to get training data or empty string ('') if training
            data are local.

    Returns:
        Primives-only dictionary describing the model trained and the evaluation outcome.
    """
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
                'skew': get_abs_diff(x[0][2], x[1][2]),
                'kurtosis': get_abs_diff(x[0][3], x[1][3]),
                'weight': x[2]
            }, paired_flat)
            paired_parsed_realized = list(paired_parsed)

            weight_sum = sum(map(lambda x: x['weight'], paired_parsed_realized))

            def get_mean(key):
                return sum(map(
                    lambda x: x[key] * x['weight'],
                    paired_parsed_realized
                )) / weight_sum

            return {
                'mean': get_mean('mean'),
                'std': get_mean('std'),
                'skew': get_mean('skew'),
                'kurtosis': get_mean('kurtosis')
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
            'trainSkew': train_errors['skew'],
            'trainKurtosis': train_errors['kurtosis'],
            'validMean': valid_errors['mean'],
            'validStd': valid_errors['std'],
            'validSkew': valid_errors['skew'],
            'validKurtosis': valid_errors['kurtosis'],
            'testMean': test_errors['mean'],
            'testStd': test_errors['std'],
            'testSkew': test_errors['skew'],
            'testKurtosis': test_errors['kurtosis']
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
    """Upload the model training frame to distributed computing-friendly storage.

    Upload the model training frame to distributed computing-friendly storage or, if remote
    distributed computing is disabled (AWS_ACCESS_KEY or AWS_ACCESS_SECRET env vars are blank),
    make a copy of the data to be used for local training.
    """

    def requires(self):
        """Indicate that the training frame is required before model training can be performed.

        Returns:
            NormalizeHistoricTrainingFrameTask
        """
        return normalize_tasks.NormalizeHistoricTrainingFrameTask()

    def output(self):
        """Indicate where the confirmation of the upload should be written.

        Returns:
            LocalTarget where the status update sould be written.
        """
        return luigi.LocalTarget(const.get_file_location('upload_historic_confirm.txt'))

    def run(self):
        """Perform the upload or, if local, copy."""
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
    """Template task for a sweep operation.

    Abstract base class (template class) for a model sweep  which tries multiple configurations and
    records the evaluative results. This is performed as a grid search.
    """

    def requires(self):
        """Require that the training frame and cluster (or local distribution) be available.

        Returns:
            UploadHistoricTrainingFrame and StartClusterTask
        """
        return {
            'upload': UploadHistoricTrainingFrame(),
            'cluster': cluster_tasks.StartClusterTask()
        }

    def output(self):
        """Indicate where the sweep results should be recorded.

        Returns:
            LocalTarget at which the CSV summary of the sweep should be written.
        """
        return luigi.LocalTarget(const.get_file_location(self.get_filename()))

    def run(self):
        """Perform the model sweep."""
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
        """Get the filename where the results should be written within the workspace.

        Returns:
            Filename (not full path) of the file to be written.
        """
        raise NotImplementedError('Must use implementor.')

    def get_num_layers(self):
        """Get list of number of layer options to include the sweep.

        Returns:
            List of integers. Behavior not defined if an option is not between 1 and 6.
        """
        raise NotImplementedError('Must use implementor.')

    def get_l2_regs(self):
        """Get the list of L2 strengths to include the sweep.

        Returns:
            List of float. Behavior not defined if an option is not between 0 and 1.
        """
        raise NotImplementedError('Must use implementor.')

    def get_dropouts(self):
        """Get the list of dropout rates to include the sweep.

        Returns:
            List of float. Behavior not defined if an option is not between 0 and 1.
        """
        raise NotImplementedError('Must use implementor.')

    def get_blocks(self):
        """Get the variables to exclude from inputs.

        Returns:
            List of strings where the 'all attrs' option does not exclude any variables. Behavior
            not defined if unknown fields provided.
        """
        raise NotImplementedError('Must use implementor.')

    def get_max_workers(self):
        """Get the maximum number of workers to allow if using remote distribution.

        Returns:
            The maximum number of workers to allow. Ignored if using local distribution.
        """
        raise NotImplementedError('Must use implementor.')

    def get_allow_counts(self):
        """Get options for allowing / not allowing sample count information to be used as an input.

        Returns:
            List of boolean values (True, False) allowed.
        """
        raise NotImplementedError('Must use implementor.')


class SweepTask(SweepTemplateTask):
    """Perform the main sweep."""

    def get_filename(self):
        """Get the filename where the results should be written within the workspace.

        Returns:
            Filename (not full path) of the file to be written.
        """
        return 'sweep.csv'

    def get_num_layers(self):
        """Get list of number of layer options to include the sweep.

        Returns:
            List of integers. Behavior not defined if an option is not between 1 and 6.
        """
        return DEFAULT_NUM_LAYERS

    def get_l2_regs(self):
        """Get the list of L2 strengths to include the sweep.

        Returns:
            List of float. Behavior not defined if an option is not between 0 and 1.
        """
        return DEFAULT_REGULARIZATION

    def get_dropouts(self):
        """Get the list of dropout rates to include the sweep.

        Returns:
            List of float. Behavior not defined if an option is not between 0 and 1.
        """
        return DEFAULT_DROPOUT

    def get_blocks(self):
        """Get the variables to exclude from inputs.

        Returns:
            List of strings where the 'all attrs' option does not exclude any variables. Behavior
            not defined if unknown fields provided.
        """
        return BLOCKS

    def get_max_workers(self):
        """Get the maximum number of workers to allow if using remote distribution.

        Returns:
            The maximum number of workers to allow. Ignored if using local distribution.
        """
        return 500

    def get_allow_counts(self):
        """Get options for allowing / not allowing sample count information to be used as an input.

        Returns:
            List of boolean values (True, False) allowed.
        """
        return [True, False] if const.INCLUDE_COUNT_IN_MODEL else [True]


class SweepExtendedTask(SweepTemplateTask):
    """Sweep parameters unlikely to be chosen but informative."""

    def get_filename(self):
        """Get the filename where the results should be written within the workspace.

        Returns:
            Filename (not full path) of the file to be written.
        """
        return 'sweep_extended.csv'

    def get_num_layers(self):
        """Get list of number of layer options to include the sweep.

        Returns:
            List of integers. Behavior not defined if an option is not between 1 and 6.
        """
        return DEFAULT_NUM_LAYERS

    def get_l2_regs(self):
        """Get the list of L2 strengths to include the sweep.

        Returns:
            List of float. Behavior not defined if an option is not between 0 and 1.
        """
        return DEFAULT_REGULARIZATION

    def get_dropouts(self):
        """Get the list of dropout rates to include the sweep.

        Returns:
            List of float. Behavior not defined if an option is not between 0 and 1.
        """
        return DEFAULT_DROPOUT

    def get_blocks(self):
        """Get the variables to exclude from inputs.

        Returns:
            List of strings where the 'all attrs' option does not exclude any variables. Behavior
            not defined if unknown fields provided.
        """
        return BLOCKS_EXTENDED

    def get_max_workers(self):
        """Get the maximum number of workers to allow if using remote distribution.

        Returns:
            The maximum number of workers to allow. Ignored if using local distribution.
        """
        return 500

    def get_allow_counts(self):
        """Get options for allowing / not allowing sample count information to be used as an input.

        Returns:
            List of boolean values (True, False) allowed.
        """
        return [True]
