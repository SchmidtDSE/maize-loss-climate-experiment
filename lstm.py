"""Tasks to train and evaluate LSTM models for yield prediction.

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
import numpy
import pandas

import cluster_tasks
import const
import normalize_tasks
import training_tasks


class SortInputDataForLstmTask(luigi.Task):
    """Sort input data by year and geohash for LSTM processing."""

    def requires(self):
        """Require normalized training data."""
        return normalize_tasks.NormalizeHistoricTrainingFrameTask()

    def output(self):
        """Indicate where sorted data should be written."""
        return luigi.LocalTarget(const.get_file_location('sorted_historic_training.csv'))

    def run(self):
        """Sort the data by year and geohash."""
        frame = pandas.read_csv(self.input().path)
        frame_sorted = frame.sort_values(['year', 'geohash'])
        frame_sorted.to_csv(self.output().path, index=False)


class UploadSortedTrainingFrame(luigi.Task):
    """Upload the sorted model training frame."""

    def requires(self):
        """Require sorted training data."""
        return SortInputDataForLstmTask()

    def output(self):
        """Indicate where upload confirmation should be written."""
        path = const.get_file_location('upload_historic_lstm_confirm.txt')
        return luigi.LocalTarget(path)

    def run(self):
        """Perform the upload or local copy."""
        access_key = os.environ.get('AWS_ACCESS_KEY', '')
        access_secret = os.environ.get('AWS_ACCESS_SECRET', '')

        using_local = access_key == '' or access_secret == ''
        if using_local:
            input_path = self.input().path
            output_path = os.path.join(const.BUCKET_OR_DIR, 'historic_training_lstm.csv')
            shutil.copy2(input_path, output_path)
        else:
            s3 = boto3.client(
                's3',
                aws_access_key_id=access_key,
                aws_secret_access_key=access_secret
            )
            s3.upload_file(
                self.input().path,
                const.BUCKET_OR_DIR,
                'historic_training_lstm.csv'
            )

        with self.output().open('w') as f:
            f.write('success')


def build_model(num_layers, num_inputs, l2_reg, dropout, learning_rate=const.LEARNING_RATE):
    """Build an LSTM-based model.

    Args:
        num_layers: The number of LSTM layers to stack.
        num_inputs: The number of input features.
        l2_reg: L2 regularization strength.
        dropout: Dropout rate for regularization.
        learning_rate: Learning rate for optimization.

    Returns:
        Untrained keras model for time series prediction.
    """
    import keras
    import keras.optimizers

    model = keras.Sequential()
    model.add(keras.Input(shape=(2, num_inputs)))  # Looking back 1 year (2 timepoints)
    model.add(keras.layers.Normalization())

    if l2_reg == 0:
        build_layer = lambda x: keras.layers.LSTM(x, return_sequences=True)
    else:
        build_layer = lambda x: keras.layers.LSTM(
            x,
            return_sequences=True,
            kernel_regularizer=keras.regularizers.L2(l2_reg)
        )

    layers = [
        build_layer(512),
        build_layer(256),
        build_layer(128),
        build_layer(64),
        build_layer(32),
        build_layer(8)
    ][-num_layers:]

    for i, layer in enumerate(layers):
        model.add(layer)
        if dropout > 0:
            model.add(keras.layers.Dropout(dropout))
        if i == len(layers) - 1:
            # Last layer should not return sequences
            model.layers[-2].return_sequences = False

    model.add(keras.layers.Dense(2, activation='linear'))

    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])

    if const.MODEL_TRANSFORM:
        return training_tasks.TransformModel(model) 
    else:
        return model


OUTPUT_ATTRS = training_tasks.OUTPUT_ATTRS
BLOCKED_ATTRS = training_tasks.BLOCKED_ATTRS


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


def try_model(access_key, secret_key, num_layers, l2_reg, dropout, bucket_name, filename,
    additional_block, allow_count, seed=12345, output_attrs=training_tasks.OUTPUT_ATTRS,
    epochs=const.EPOCHS, blocked_attrs=training_tasks.BLOCKED_ATTRS):
    """Try building and training an LSTM model."""
    import random
    import pandas

    random.seed(seed)

    using_local = access_key == '' or secret_key == ''
    temp_file_path = os.path.join(bucket_name, filename) if using_local else '/tmp/' + filename

    if not using_local and not os.path.isfile(temp_file_path):
        s3 = boto3.resource(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        s3_object = s3.Object(bucket_name, filename)
        s3_object.download_file(temp_file_path)

    def process_data(frame):
        input_attrs = get_input_attrs(additional_block, allow_count)
        
        # Group by geohash to create sequences
        groups = frame.groupby('geohash')
        inputs = []
        outputs = []
        
        for _, group in groups:
            group_sorted = group.sort_values('year')
            for i in range(len(group_sorted) - 1):
                sequence = group_sorted.iloc[i:i+2][input_attrs].values.astype('float32')
                inputs.append(sequence)
                outputs.append(group_sorted.iloc[i+1][output_attrs].values.astype('float32'))
        
        return numpy.array(inputs, dtype='float32'), numpy.array(y, dtype='float32')

    frame = pandas.read_csv(temp_file_path)
    frame['setAssign'] = frame['year'].apply(
        lambda x: 'train' if x < 2013 else ('test' if x % 2 == 0 else 'valid')
    )

    data_splits = {
        'train': process_data(frame[frame['setAssign'] == 'train']),
        'valid': process_data(frame[frame['setAssign'] == 'valid']),
        'test': process_data(frame[frame['setAssign'] == 'test'])
    }

    model = build_model(num_layers, len(get_input_attrs(additional_block, allow_count)), l2_reg, dropout)
    
    # Train model
    model.fit(
        data_splits['train'][0],
        data_splits['train'][1],
        epochs=epochs,
        verbose=None
    )

    # Evaluate model
    def evaluate_split(inputs, outputs):
        predictions = model.predict(inputs, verbose=None)
        mean_errors = numpy.abs(predictions[:, 0] - outputs[:, 0])
        std_errors = numpy.abs(predictions[:, 1] - outputs[:, 1])
        return {
            'mean': numpy.mean(mean_errors),
            'std': numpy.mean(std_errors)
        }

    results = {
        'train': evaluate_split(data_splits['train'][0], data_splits['train'][1]),
        'valid': evaluate_split(data_splits['valid'][0], data_splits['valid'][1]),
        'test': evaluate_split(data_splits['test'][0], data_splits['test'][1])
    }

    return {
        'block': additional_block,
        'layers': num_layers,
        'l2Reg': l2_reg,
        'dropout': dropout,
        'allowCount': allow_count,
        'trainMean': results['train']['mean'],
        'trainStd': results['train']['std'],
        'validMean': results['valid']['mean'],
        'validStd': results['valid']['std'],
        'testMean': results['test']['mean'],
        'testStd': results['test']['std']
    }


class LstmSweepTemplateTask(luigi.Task):
    """Template task for LSTM model parameter sweep."""

    def requires(self):
        """Require training frame upload and cluster availability."""
        return {
            'upload': UploadSortedTrainingFrame(),
            'cluster': cluster_tasks.StartClusterTask()
        }

    def output(self):
        """Indicate where sweep results should be recorded."""
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
                'historic_training_lstm.csv',
                x[3],
                x[4]
            ),
            combinations_realized
        )

        with self.output().open('w') as f:
            writer = csv.DictWriter(f, fieldnames=training_tasks.OUTPUT_FIELDS)
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

class LstmSweepTask(LstmSweepTemplateTask):
    """Perform main LSTM model sweep."""

    def get_filename(self):
        return 'lstm_sweep.csv'

    def get_num_layers(self):
        return training_tasks.DEFAULT_NUM_LAYERS

    def get_l2_regs(self):
        return training_tasks.DEFAULT_REGULARIZATION

    def get_dropouts(self):
        return training_tasks.DEFAULT_DROPOUT

    def get_blocks(self):
        return training_tasks.BLOCKS

    def get_max_workers(self):
        return 450

    def get_allow_counts(self):
        return [True, False] if const.INCLUDE_COUNT_IN_MODEL else [True]
