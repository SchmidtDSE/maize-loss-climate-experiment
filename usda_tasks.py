"""Tasks for summarizing USDA metrics.

Tasks for summarizing USDA metrics for the purposes of the disscussion as supplemental results which
are highly summarized.

License: BSD
"""
import csv
import io
import itertools
import zipfile

import luigi
import requests
import toolz.itertoolz

import cluster_tasks
import const
import sim_tasks

ZIP_URL_TEMPLATE = 'https://www.rma.usda.gov/sites/default/files/information-tools/sobcov_%d.zip'
FIELDS = [
    'year',
    'state',
    'stateAbbrevation',
    'countyCode',
    'countyName',
    'commodityCode',
    'commodityName',
    'insurancePlanCode',
    'insuranceName',
    'coverage',
    'deliveryType',
    'coverageLevel',
    'policiesSoldCount',
    'policiesPremiumCount',
    'policiesIdemnifiedCount',
    'unitsPremiumCount',
    'unitsIndemnifiedCount',
    'quantityType',
    'netReported',
    'endoresed',
    'liabilityAmounnt',
    'premium',
    'subsidy',
    'statePrivateSubsidy',
    'otherSubsidy',
    'efaDiscount',
    'idemnityAmount',
    'lossRatio'
]

FINAL_FIELDS = ['year', 'actualCount', 'actualClaims', 'actualLossRatio', 'simCount', 'simClaims']


class SummarizeUsdaYearCountyTask(luigi.Task):
    """Summarize at the county level per year per program."""

    def output(self):
        """Indicate where the county per year per program output should be written.

        Returns:
            Location where the output CSV file should be written.
        """
        return luigi.LocalTarget(const.get_file_location('usda_post_summary.csv'))

    def run(self):
        """Summarize the USDA dataset for corn."""
        nested = map(lambda x: self._process_year(x), const.YEARS)
        flattened = itertools.chain(*nested)

        with self.output().open('w') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['year', 'county', 'program', 'count', 'indemnified', 'lossRatio']
            )
            writer.writeheader()
            writer.writerows(flattened)
    
    def _process_year(self, year):
        """Summarize a year of summary of business records.

        Args:
            year: The integer year to process.
        Returns:
            Summary rows.
        """
        url = ZIP_URL_TEMPLATE % year
        response = requests.get(url)

        assert response.status_code < 400

        zip_data = io.BytesIO(response.content)

        def make_row(fields):
            fields_stripped = map(lambda x: x.strip(), fields)
            fields_zipped = zip(FIELDS, fields_stripped)
            return dict(fields_zipped)

        def simplify_row(raw_row):
            return {
                'year': year,
                'county': '%s, %s' % (raw_row['countyName'], raw_row['stateAbbrevation']),
                'count': int(raw_row['unitsPremiumCount']),
                'indemnified': int(raw_row['unitsIndemnifiedCount']),
                'lossRatio': float(raw_row['lossRatio']),
                'program': raw_row['insuranceName']
            }

        def combine_rows(row_a, row_b):
            assert row_a['year'] == row_b['year']
            assert row_a['county'] == row_b['county']
            assert row_a['program'] == row_b['program']

            combined_count = row_a['count'] + row_b['count']
            combined_indemnified = row_a['indemnified'] + row_b['indemnified']

            loss_ratio_weighted_a = row_a['count'] * row_a['lossRatio']
            loss_ratio_weighted_b = row_b['count'] * row_b['lossRatio']
            loss_ratio = (loss_ratio_weighted_a + loss_ratio_weighted_b) / combined_count

            return {
                'year': row_a['year'],
                'county': row_a['county'],
                'program': row_a['program'],
                'count': combined_count,
                'indemnified': combined_indemnified,
                'lossRatio': loss_ratio
            }

        with zipfile.ZipFile(zip_data) as zip_file:
            csv_filename = zip_file.namelist()[0]

            with zip_file.open(csv_filename) as csv_file:
                csv_text = io.StringIO(csv_file.read().decode('utf-8'))
                reader = csv.reader(csv_text, delimiter='|')

                rows = map(make_row, reader)
                maize_rows = filter(
                    lambda x: x['commodityName'].lower() in ['corn', 'sweet corn'],
                    rows
                )
                in_scope_rows = filter(
                    lambda x: x['insuranceName'].lower() in ['yp', 'rp', 'aph'],
                    maize_rows
                )
                simplified_rows = map(simplify_row, in_scope_rows)
                simplified_rows_valid = filter(lambda x: x['count'] > 0, simplified_rows)

                reduced_rows_keyed = toolz.itertoolz.reduceby(
                    lambda x: x['county'] + '\t' + x['program'],
                    combine_rows,
                    simplified_rows_valid
                )
                reduced_rows = reduced_rows_keyed.values()

                if len(reduced_rows) == 0:
                    raise RuntimeError('No records found on %s.' % url)

                return reduced_rows


class SummarizeYearlySimClaims(luigi.Task):

    def requires(self):
        return sim_tasks.ExecuteSimulationTasksHistoricPredictedTask()

    def output(self):
        return luigi.LocalTarget(const.get_file_location('usda_post_sim_yearly_summary.csv'))

    def run(self):

        def make_row(target):
            return {
                'year': int(target['year']),
                'unitSize': int(target['unitSize']),
                'predictedClaims': float(target['predictedClaims'])
            }

        def combine_rows(row_a, row_b):
            assert row_a['year'] == row_b['year']

            size_a = row_a['unitSize']
            size_b = row_b['unitSize']
            combined_size = size_a + size_b

            weighted_a = size_a * row_a['predictedClaims']
            weighted_b = size_b * row_b['predictedClaims']
            new_claims = (weighted_a + weighted_b) / combined_size

            return {
                'year': row_a['year'],
                'unitSize': combined_size,
                'predictedClaims': new_claims
            }

        with self.input().open('r') as f:
            raw_rows = csv.DictReader(f)
            with_baseline = filter(lambda x: x['offsetBaseline'] == 'always', raw_rows)
            with_condition = filter(lambda x: x['condition'] == 'historic', with_baseline)
            with_threshold = filter(
                lambda x: abs(float(x['threshold']) - 0.25) < 0.00001,
                with_condition
            )
            with_std = filter(lambda x: abs(int(x['stdMult']) - 1) < 0.00001, with_threshold)
            with_geohash = filter(lambda x: int(x['geohashSimSize']) == 4, with_std)
            rows = map(make_row, with_geohash)

            reduced_rows_keyed = toolz.itertoolz.reduceby(
                lambda x: x['year'],
                combine_rows,
                rows
            )
            reduced_rows = reduced_rows_keyed.values()

            with self.output().open('w') as f:
                writer = csv.DictWriter(f, fieldnames=['year', 'unitSize', 'predictedClaims'])
                writer.writeheader()
                writer.writerows(reduced_rows)


class SummarizeYearlyActualClaims(luigi.Task):

    def requires(self):
        return SummarizeUsdaYearCountyTask()

    def output(self):
        return luigi.LocalTarget(const.get_file_location('usda_post_actual_yearly_summary.csv'))

    def run(self):

        def make_row(target):
            return {
                'year': int(target['year']),
                'count': int(target['count']),
                'indemnified': int(target['indemnified']),
                'lossRatio': float(target['lossRatio'])
            }

        def combine_rows(row_a, row_b):
            assert row_a['year'] == row_b['year']

            count_a = row_a['count']
            count_b = row_b['count']
            combined_count = count_a + count_b

            indemnified_a = row_a['indemnified']
            indemnified_b = row_b['indemnified']
            combined_indemnified = indemnified_a + indemnified_b

            weighted_a = count_a * row_a['lossRatio']
            weighted_b = count_b * row_b['lossRatio']
            new_loss_ratio = (weighted_a + weighted_b) / combined_count

            return {
                'year': row_a['year'],
                'count': combined_count,
                'indemnified': combined_indemnified,
                'lossRatio': new_loss_ratio
            }

        with self.input().open('r') as f:
            raw_rows = csv.DictReader(f)
            rows = map(make_row, raw_rows)

            reduced_rows_keyed = toolz.itertoolz.reduceby(
                lambda x: x['year'],
                combine_rows,
                rows
            )
            reduced_rows = reduced_rows_keyed.values()

            with self.output().open('w') as f:
                writer = csv.DictWriter(f, fieldnames=['year', 'count', 'indemnified', 'lossRatio'])
                writer.writeheader()
                writer.writerows(reduced_rows)


class CombineYearlyActualClaims(luigi.Task):

    def requires(self):
        return {
            'sim': SummarizeYearlySimClaims(),
            'actual': SummarizeYearlyActualClaims()
        }

    def output(self):
        return luigi.LocalTarget(const.get_file_location('usda_post_combined_yearly_summary.csv'))

    def run(self):
        sim_keyed = self._get_keyed('sim')
        actual_keyed = self._get_keyed('actual')

        sim_keys = set(sim_keyed.keys())
        actual_keys = set(actual_keyed.keys())
        combined_keys = sim_keys.intersection(actual_keys)

        def make_output_row(key):
            sim_row = sim_keyed[key]
            actual_row = actual_keyed[key]
            return {
                'year': sim_row['year'],
                'actualCount': actual_row['count'],
                'actualClaims': float(actual_row['indemnified']) / actual_row(sim_row['count']),
                'actualLossRatio': actual_row['lossRatio'],
                'simCount': sim_row['unitSize'],
                'simClaims': sim_row['predictedClaims']
            }

        output_rows = map(make_output_row, combined_keys)

        with self.output().open('w') as f:
            writer = csv.DictWriter(f, fieldnames=FINAL_FIELDS)
            writer.writeheader()
            writer.writerows(output_rows)

    def _get_indexed(self, file_key):
        with self.input()[file_key] as f:
            records = csv.DictReader(f)
            records_tuple = map(lambda x: (int(x['year'], x)), records)
            return dict(records_tuple)

