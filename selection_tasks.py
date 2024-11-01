"""Tasks which examines / selects from candidate sweep models and run post-hoc tests.

License:
    BSD
"""
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
    """Select the leading preferred model from the sweep."""

    def requires(self):
        """Require that the sweep be completed before choosing a perferred model.

        Returns:
            SweepTask
        """
        return training_tasks.SweepTask()

    def output(self):
        """Determine where the preferred configuration information should be written.

        Returns:
            LocalTarget at which model configuration information should be written as JSON.
        """
        return luigi.LocalTarget(const.get_file_location('selected_config.json'))

    def run(self):
        """Choose a preferred configuration."""
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
        """Parse a single input row describing a single model candidate.

        Args:
            row: The raw row describing a single model candidate to parse.

        Returns:
            Parsed row describing a single candidate model.
        """
        fields = row.keys()
        fields_to_interpret = filter(lambda x: x not in STR_META_ATTRS, fields)

        for field in fields_to_interpret:
            row[field] = float(row[field])

        return row


class PostHocTestRawDataTemplateTask(luigi.Task):
    """Template test for a post hoc task."""

    def requires(self):
        """Get the dependencies for this task.

        Indicate that the sweep selection task needs to be completed and request access to the
        historic training frame.

        Returns:
            SelectConfigurationTask and NormalizeHistoricTrainingFrameTask
        """
        return {
            'configuration': SelectConfigurationTask(),
            'training': normalize_tasks.NormalizeHistoricTrainingFrameTask()
        }

    def output(self):
        """Get the location at which the post-hoc test results should be written.

        Returns:
            LocalTarget at which results should be written.
        """
        return luigi.LocalTarget(const.get_file_location(self.get_filename()))

    def run(self):
        """Execute this post-hoc test."""
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
            epochs=const.EPOCHS,
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
        """Determine into which test (train, test) a record should be assigned.

        Args:
            record: The record to assign.

        Returns:
            String indicating where the record should be assigned.
        """
        raise NotImplementedError('Must use implementor.')

    def get_filename(self):
        """Get the filename at which results should be written inside the workspace.

        Returns:
            Filename at which results should be written.
        """
        raise NotImplementedError('Must use implementor.')

    def get_output_cols(self):
        """Get the columns expected in output records.

        Returns:
            List of string.
        """
        raise NotImplementedError('Must use implementor.')

    def output_test_only(self):
        """Determine if this task should ouput results on the test set.

        Returns:
            True if only test records should be reported or false if all records should be reported.
        """
        raise NotImplementedError('Must use implementor.')

    def prep_set_assign(self, frame):
        """Prepare internal data structures for set assignment.

        Args:
            frame: The input frame that will be later assigned.
        """
        pass


class PostHocTestRawDataTemporalResidualsTask(PostHocTestRawDataTemplateTask):
    """Post-hoc test which tests a model's ability to predict into the future.

    Post-hoc test which tests a model's ability to predict into the future, reporting only on test
    set (used for Monte Carlo sampling).
    """

    def get_set_assign(self, record):
        """Assign records based on their year.

        Args:
            record: The record to assign.

        Returns:
            The set assignment as a string.
        """
        return 'train' if record['year'] < 2014 else 'test'

    def get_filename(self):
        """Get the filename at which results should be written inside the workspace.

        Returns:
            Filename at which results should be written.
        """
        return 'post_hoc_temporal.csv'

    def get_output_cols(self):
        """Get the columns expected in output records.

        Returns:
            List of string.
        """
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
        """Determine if this task should ouput results on the test set.

        Returns:
            True if only test records should be reported or false if all records should be reported.
        """
        return True


class PostHocTestRawDataClimateCountTask(PostHocTestRawDataTemplateTask):
    """Post-hoc test which tests a model's ability to predict with out of sample climate data."""

    def get_set_assign(self, record):
        """Assign records based on their year.

        Args:
            record: The record to assign.

        Returns:
            The set assignment as a string.
        """
        return 'test' if record['year'] == 2012 else 'train'

    def get_filename(self):
        """Get the filename at which results should be written inside the workspace.

        Returns:
            Filename at which results should be written.
        """
        return 'post_hoc_climate_count.csv'

    def get_output_cols(self):
        """Get the columns expected in output records.

        Returns:
            List of string.
        """
        # Weighting by geohash
        return [
            'geohash',
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
        """Determine if this task should ouput results on the test set.

        Returns:
            True if only test records should be reported or false if all records should be reported.
        """
        return False


class PostHocTestRawDataTemporalCountTask(PostHocTestRawDataTemplateTask):
    """Post-hoc test which tests a model's ability to predict into the future.

    Post-hoc test which tests a model's ability to predict into the future, reporting on all data
    for informational purposes.
    """

    def get_set_assign(self, record):
        """Assign records based on their year.

        Args:
            record: The record to assign.

        Returns:
            The set assignment as a string.
        """
        return 'train' if record['year'] < 2014 else 'test'

    def get_filename(self):
        """Get the filename at which results should be written inside the workspace.

        Returns:
            Filename at which results should be written.
        """
        return 'post_hoc_temporal_with_count.csv'

    def get_output_cols(self):
        """Get the columns expected in output records.

        Returns:
            List of string.
        """
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
        """Determine if this task should ouput results on the test set.

        Returns:
            True if only test records should be reported or false if all records should be reported.
        """
        return False


class PostHocTestRawDataRetrainCountTask(PostHocTestRawDataTemplateTask):
    """Post-hoc test for test set from sweep with expanded training.

    Post-hoc test that calculates test set performance after retraining on train and validation from
    the sweep.
    """

    def get_set_assign(self, record):
        """Apply a set assignment consistent with sweep task.

        Args:
            record: The record to assign.

        Returns:
            The set assignment.
        """
        return 'test' if record['year'] in [2013, 2015] else 'train'

    def get_filename(self):
        """Get the filename at which results should be written inside the workspace.

        Returns:
            Filename at which results should be written.
        """
        return 'post_hoc_retrain_with_count.csv'

    def get_output_cols(self):
        """Get the columns expected in output records.

        Returns:
            List of string.
        """
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
        """Determine if this task should ouput results on the test set.

        Returns:
            True if only test records should be reported or false if all records should be reported.
        """
        return False


class PostHocTestRawDataRandomCountTask(PostHocTestRawDataTemplateTask):

    def get_set_assign(self, record):
        """Randomly assign a record to train or test set.

        Args:
            record: The record to assign.

        Returns:
            The set assignment as string.
        """
        return random.choice(['train', 'train', 'train', 'test'])

    def get_filename(self):
        """Get the filename at which results should be written inside the workspace.

        Returns:
            Filename at which results should be written.
        """
        return 'post_hoc_random_with_count.csv'

    def get_output_cols(self):
        """Get the columns expected in output records.

        Returns:
            List of string.
        """
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
        """Determine if this task should ouput results on the test set.

        Returns:
            True if only test records should be reported or false if all records should be reported.
        """
        return False


class PostHocTestRawDataSpatialCountTask(PostHocTestRawDataTemplateTask):

    def prep_set_assign(self, frame):
        """Prepare internal data structures for set assignment.

        Args:
            frame: The input frame that will be later assigned.
        """
        unique_geohashes = set(frame['geohash'].apply(lambda x: x[:3]).unique())
        unique_geohashes_assigned = map(
            lambda x: (x, random.choice(['train', 'train', 'train', 'test'])),
            unique_geohashes
        )
        self._geohash_assignments = dict(unique_geohashes_assigned)

    def get_set_assign(self, record):
        """Assign this record to a set based on the 3 character geohash prefix.

        Args:
            record: The record to assign.

        Returns:
            Set assignment as string.
        """
        return self._geohash_assignments[record['geohash'][:3]]

    def get_filename(self):
        """Get the filename at which results should be written inside the workspace.

        Returns:
            Filename at which results should be written.
        """
        return 'post_hoc_spatial_with_count.csv'

    def get_output_cols(self):
        """Get the columns expected in output records.

        Returns:
            List of string.
        """
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
        """Determine if this task should ouput results on the test set.

        Returns:
            True if only test records should be reported or false if all records should be reported.
        """
        return False


class TrainFullModel(luigi.Task):
    """Task which retrains the model using preferred configuration on all available data."""

    def requires(self):
        """Get the dependencies for this task.

        Indicate that the sweep selection task needs to be completed and request access to the
        historic training frame.

        Returns:
            SelectConfigurationTask and NormalizeHistoricTrainingFrameTask
        """
        return {
            'configuration': SelectConfigurationTask(),
            'training': normalize_tasks.NormalizeHistoricTrainingFrameTask()
        }

    def output(self):
        """Determine where the final model should be written (as .keras).

        Returns:
            LocalTarget at which to serialize the model.
        """
        return luigi.LocalTarget(const.get_file_location('model.keras'))

    def run(self):
        """Train the final model."""
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
            epochs=const.EPOCHS,
            verbose=None,
            sample_weight=input_frame[const.SAMPLE_WEIGHT_ATTR]
        )

        model.save(self.output().path)
