import csv
import json

import luigi

import const
import distribution_struct
import preprocess_combine_tasks


def parse_row(row):
    for field in row:
        if field in const.TRAINING_STR_FIELDS:
            row[field] = row[field]
        elif field in const.TRAINING_INT_FIELDS:
            row[field] = int(row[field])
        else:
            row[field] = float(row[field])

    return row


class GetInputDistributionsTaskTemplate(luigi.Task):

    def requires(self):
        preprocess_combine_tasks.CombineHistoricPreprocessTask()

    def output(self):
        return luigi.LocalTarget(const.get_file_location('historic_z.json'))

    def run(self):
        fields_to_process = set(const.TRAINING_FRAME_ATTRS) - set(const.NON_Z_FIELDS)
        accumulators = dict(map(
            lambda x: (x, distribution_struct.WelfordAccumulator()),
            fields_to_process
        ))

        with self.input().open('r') as f:
            input_records = csv.DictReader(f)

            for row in input_records:
                for field in fields_to_process:
                    value = float(row[field])
                    accumulators[field].add(value)

        output_rows = map(
            lambda x: self._serialize_accumulator(x[0], x[1]),
            accumulators.items()
        )

        with self.output().open('w') as f:
            writer = csv.DictWriter(f)
            writer.writeheader()
            writer.writerows(output_rows)

    def _serialize_accumulator(self, field, accumulator):
        return {
            'field': field,
            'mean': accumulator.get_mean(),
            'std': accumulator.get_std()
        }


class NormalizeTrainingFrame(luigi.Task):

    def requires(self):
        return {
            'distributions': GetInputDistributionsTaskTemplate(),
            'target': self.get_target()
        }

    def output(self):
        return luigi.LocalTarget(const.get_file_location(self.get_filename()))

    def run(self):
        with self.input()['distributions'].open('r') as f:
            distributions = json.load(f)

        with self.input()['target'].open('r') as f_in:
            reader = csv.DictReader(f)

            rows = map(lambda x: self._parse_row(x), reader)
            rows_augmented = map(lambda x: self._set_aside_attrs(x), rows)
            rows_with_response = map(lambda x: self._transform_response(x), rows_augmented)
            rows_with_z = map(lambda x: self._transform_z(x), rows_with_response)
            rows_standardized = map(lambda x: self._standardize_row(x), rows_with_z)

            with self.output().open('w') as f_out:
                writer = csv.DictWriter(f)
                writer.writeheader(const.TRAINING_FRAME_ATTRS)
                writer.writerows(rows_standardized)

    def get_target(self):
        raise NotImplementedError('Use implementor.')

    def get_filename(self):
        raise NotImplementedError('Use implementor.')

    def _parse_row(self, row):
        return parse_row(row)

    def _set_aside_attrs(self, row):
        row['baselineYieldMeanOriginal'] = row['baselineYieldMean']
        row['baselineYieldStdOriginal'] = row['baselineYieldStd']
        return row

    def _transform_response(self, row):
        get_percent_change = lambda (start, end): (end - start) / start

        baseline_mean = row['baselineYieldMeanOriginal']
        baseline_std = row['baselineYieldStdOriginal']

        year_mean = row['yieldMean']
        year_std = row['yieldStd']
        
        row['yieldMean'] = get_percent_change(year_mean, baseline_mean)
        row['yieldStd'] = get_percent_change(year_std, baseline_std)

        return row

    def _transform_z(self, row, distributions):
        for field in distributions:
            original_value = row[field]
            
            distribution = distributions[field]
            mean = distributions['mean']
            std = distributions['std']
            
            row[field] = (row[field] - mean) / std

        return row

    def _standardize_row(self, row):
        return dict(map(lambda x: (x, row[x]), TRAINING_FRAME_ATTRS))


class NormalizeHistoricTrainingFrame(NormalizeTrainingFrame):

    def get_target(self):
        return preprocess_combine_tasks.CombineHistoricPreprocessTask()

    def get_filename(self):
        return 'historic_normalized.csv'
