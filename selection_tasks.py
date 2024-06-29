import csv
import json

import keras
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
CONSTRAINED_REG = [0, 1]
CONSTRAINED_ATTRS = ['all attrs', 'year']


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

        mean_mean = mean_accumulator.get_mean()
        mean_std = mean_accumulator.get_std()
        std_mean = std_accumulator.get_mean()
        std_std = std_accumulator.get_std()
        
        def score_option(option):
            mean_z = option['validMean']  # (option['validMean'] - mean_mean) / mean_std
            std_z = option['validStd']  # (option['validStd'] - std_mean) / std_std
            return mean_z + std_z / 2

        unconstrained_selection_row = min(rows, key=score_option)

        def get_regularization_ok(target):
            return target >= CONSTRAINED_REG[0] and target <= CONSTRAINED_REG[1]

        constrained_candidates = filter(
            lambda x: (
                x['layers'] in CONSTRAINED_LAYERS
                and x['block'] in CONSTRAINED_ATTRS
                and get_regularization_ok(x['l2Reg'])
                and get_regularization_ok(x['dropout'])
            ),
            rows
        )

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
            epochs=35,
            verbose=None,
            sample_weight=train_data[const.SAMPLE_WEIGHT_ATTR]
        )

        combined_output = model.predict(input_frame[input_attrs])
        input_frame['predictedMean'] = combined_output[:,0]
        input_frame['predictedStd'] = combined_output[:,1]
        input_frame['meanResidual'] = input_frame['predictedMean'] - input_frame['yieldMean']
        input_frame['stdResidual'] = input_frame['predictedStd'] - input_frame['yieldStd']

        test_frame = input_frame[input_frame['setAssign'] == 'test']
        test_frame[[
            'setAssign',
            'yieldMean',
            'yieldStd',
            'predictedMean',
            'predictedStd',
            'meanResidual',
            'stdResidual'
        ]].to_csv(self.output().path)

    def get_set_assign(self, record):
        raise NotImplementedError('Must use implementor.')

    def get_filename(self):
        raise NotImplementedError('Must use implementor.')


class PostHocTestRawDataTemporalTask(PostHocTestRawDataTemplateTask):

    def get_set_assign(self, record):
        return 'train' if record['year'] < 2014 else 'test'

    def get_filename(self):
        return 'post_hoc_temporal.csv'


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
            epochs=35,
            verbose=None,
            sample_weight=input_frame[const.SAMPLE_WEIGHT_ATTR]
        )

        model.save(self.output().path)
