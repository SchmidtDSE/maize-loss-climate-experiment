import csv
import json
import random

import luigi
import pandas

import const
import distribution_struct
import normalize_tasks
import training_tasks

STR_META_ATTRS = {
    'block',
    'allowCount'
}
CONSTRAINED_LAYERS = set(range(3, 7))
CONSTRAINED_ATTRS = ['all attrs']


class SelectConfigurationTask(luigi.Task):

    def requires(self):
        return training_tasks.SweepTask()

    def output(self):
        return luigi.LocalTarget(const.get_file_location('selected_config.json'))

    def run(self):
        mean_accumulator = distribution_struct.WelfordAccumulator()
        std_accumulator = distribution_struct.WelfordAccumulator()

        with self.input().open('r') as f:
            reader = csv.DictReader(f)
            rows = [self._parse_row(x) for x in reader]

        for row in rows:
            mean_accumulator.add(row['validMean'])
            std_accumulator.add(row['validStd'])

        def score_option(option):
            mean_z = option['validMean']
            std_z = option['validStd']
            return mean_z + std_z / 3

        unconstrained_selection_row = min(rows, key=score_option)

        def is_in_constrained(x):
            conditions = [
                x['layers'] in CONSTRAINED_LAYERS,
                x['block'] in CONSTRAINED_ATTRS,
                (x['l2Reg'] > 0 or x['dropout'] > 0),
                x['allowCount'].lower() == 'true'
            ]
            invalid_conditions = filter(lambda x: x is False, conditions)
            count_invalid = sum(map(lambda x: 1, invalid_conditions))
            return count_invalid == 0

        constrained_candidates = filter(is_in_constrained, rows)

        constrained_selection_row = min(constrained_candidates, key=score_option)

        with self.output().open('w') as f:
            json.dump({
                'unconstrained': unconstrained_selection_row,
                'constrained': constrained_selection_row
            }, f, indent=4)

    def _parse_row(self, row):
        fields = row.keys()
        fields_to_interpret = filter(lambda x: x not in STR_META_ATTRS, fields)

        for field in fields_to_interpret:
            row[field] = float(row[field])

        return row


class PostHocTestRawDataTemplateTask(luigi.Task):

    def requires(self):
        return {
            'configuration': SelectConfigurationTask(),
            'training': normalize_tasks.NormalizeHistoricTrainingFrameTask()
        }

    def output(self):
        return luigi.LocalTarget(const.get_file_location(self.get_filename()))

    def run(self):
        with self.input()['configuration'].open('r') as f:
            configuration = json.load(f)['constrained']

        input_frame = pandas.read_csv(self.input()['training'].path)
        self.prep_set_assign(input_frame)
        input_frame['setAssign'] = input_frame.apply(
            lambda x: self.get_set_assign(x),
            axis=1
        )

        additional_block = configuration['block']
        allow_count = configuration['allowCount'].lower() == 'true'
        num_layers = round(configuration['layers'])
        l2_reg = configuration['l2Reg']
        dropout = configuration['dropout']

        input_attrs = training_tasks.get_input_attrs(additional_block, allow_count)
        num_inputs = len(input_attrs)
        model = training_tasks.build_model(num_layers, num_inputs, l2_reg, dropout)

        train_data = input_frame[input_frame['setAssign'] == 'train']
        train_inputs = train_data[input_attrs]
        train_outputs = train_data[training_tasks.OUTPUT_ATTRS]

        model.fit(
            train_inputs,
            train_outputs,
            epochs=30,
            verbose=None,
            sample_weight=train_data[const.SAMPLE_WEIGHT_ATTR]
        )

        combined_output = model.predict(input_frame[input_attrs])
        input_frame['predictedMean'] = combined_output[:, 0]
        input_frame['predictedStd'] = combined_output[:, 1]
        input_frame['meanResidual'] = input_frame['predictedMean'] - input_frame['yieldMean']
        input_frame['stdResidual'] = input_frame['predictedStd'] - input_frame['yieldStd']

        if self.output_test_only():
            test_frame = input_frame[input_frame['setAssign'] == 'test']
            test_frame[self.get_output_cols()].to_csv(self.output().path)
        else:
            input_frame[self.get_output_cols()].to_csv(self.output().path)

    def get_set_assign(self, record):
        raise NotImplementedError('Must use implementor.')

    def get_filename(self):
        raise NotImplementedError('Must use implementor.')

    def get_output_cols(self):
        raise NotImplementedError('Must use implementor.')

    def output_test_only(self):
        raise NotImplementedError('Must use implementor.')

    def prep_set_assign(self, frame):
        pass


class PostHocTestRawDataTemporalResidualsTask(PostHocTestRawDataTemplateTask):

    def get_set_assign(self, record):
        return 'train' if record['year'] < 2014 else 'test'

    def get_filename(self):
        return 'post_hoc_temporal.csv'

    def get_output_cols(self):
        # Weighting by geohash
        return [
            'setAssign',
            'yieldMean',
            'yieldStd',
            'predictedMean',
            'predictedStd',
            'meanResidual',
            'stdResidual'
        ]

    def output_test_only(self):
        return True


class PostHocTestRawDataTemporalCountTask(PostHocTestRawDataTemplateTask):

    def get_set_assign(self, record):
        return 'train' if record['year'] < 2014 else 'test'

    def get_filename(self):
        return 'post_hoc_temporal_with_count.csv'

    def get_output_cols(self):
        # Weighting by unit
        return [
            'setAssign',
            'yieldMean',
            'yieldStd',
            'predictedMean',
            'predictedStd',
            'meanResidual',
            'stdResidual',
            'yieldObservations'
        ]

    def output_test_only(self):
        return False


class PostHocTestRawDataRetrainCountTask(PostHocTestRawDataTemplateTask):

    def get_set_assign(self, record):
        return 'test' if record['year'] in [2013, 2015] else 'train'

    def get_filename(self):
        return 'post_hoc_retrain_with_count.csv'

    def get_output_cols(self):
        # Weighting by unit
        return [
            'setAssign',
            'yieldMean',
            'yieldStd',
            'predictedMean',
            'predictedStd',
            'meanResidual',
            'stdResidual',
            'yieldObservations'
        ]

    def output_test_only(self):
        return False


class PostHocTestRawDataRandomCountTask(PostHocTestRawDataTemplateTask):

    def get_set_assign(self, record):
        return random.choice(['train', 'train', 'train', 'test'])

    def get_filename(self):
        return 'post_hoc_random_with_count.csv'

    def get_output_cols(self):
        # Weighting by unit
        return [
            'setAssign',
            'yieldMean',
            'yieldStd',
            'predictedMean',
            'predictedStd',
            'meanResidual',
            'stdResidual',
            'yieldObservations'
        ]

    def output_test_only(self):
        return False


class PostHocTestRawDataSpatialCountTask(PostHocTestRawDataTemplateTask):

    def prep_set_assign(self, frame):
        unique_geohashes = set(frame['geohash'].apply(lambda x: x[:3]).unique())
        unique_geohashes_assigned = map(
            lambda x: (x, random.choice(['train', 'train', 'train', 'test'])),
            unique_geohashes
        )
        self._geohash_assignments = dict(unique_geohashes_assigned)

    def get_set_assign(self, record):
        return self._geohash_assignments[record['geohash'][:3]]

    def get_filename(self):
        return 'post_hoc_spatial_with_count.csv'

    def get_output_cols(self):
        # Weighting by unit
        return [
            'setAssign',
            'yieldMean',
            'yieldStd',
            'predictedMean',
            'predictedStd',
            'meanResidual',
            'stdResidual',
            'yieldObservations'
        ]

    def output_test_only(self):
        return False


class TrainFullModel(luigi.Task):

    def requires(self):
        return {
            'configuration': SelectConfigurationTask(),
            'training': normalize_tasks.NormalizeHistoricTrainingFrameTask()
        }

    def output(self):
        return luigi.LocalTarget(const.get_file_location('model.keras'))

    def run(self):
        with self.input()['configuration'].open('r') as f:
            configuration = json.load(f)['constrained']

        input_frame = pandas.read_csv(self.input()['training'].path)

        additional_block = configuration['block']
        allow_count = configuration['allowCount'].lower() == 'true'
        num_layers = round(configuration['layers'])
        l2_reg = configuration['l2Reg']
        dropout = configuration['dropout']

        input_attrs = training_tasks.get_input_attrs(additional_block, allow_count)
        num_inputs = len(input_attrs)
        model = training_tasks.build_model(num_layers, num_inputs, l2_reg, dropout)

        train_inputs = input_frame[input_attrs]
        train_outputs = input_frame[training_tasks.OUTPUT_ATTRS]

        model.fit(
            train_inputs,
            train_outputs,
            epochs=30,
            verbose=None,
            sample_weight=input_frame[const.SAMPLE_WEIGHT_ATTR]
        )

        model.save(self.output().path)
