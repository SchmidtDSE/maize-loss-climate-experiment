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

import const

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


class SummarizeYearlyActualClaims(luigi.Task):
    """Summarize actual system-wide claims per year."""

    def requires(self):
        """Indicate that data were needed from the USDA to execute this task.

        Returns:
            SummarizeUsdaYearCountyTask
        """
        return SummarizeUsdaYearCountyTask()

    def output(self):
        """Indicate where the actuals CSV file should be written.

        Returns:
            The CSV file where the yearly summary for actuals should be written.
        """
        return luigi.LocalTarget(const.get_file_location('usda_post_actual_yearly_summary.csv'))

    def run(self):
        """Summarize the system-wide yearly metrics from USDA actuals."""

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

        def complete_row(target):
            return {
                'year': target['year'],
                'count': target['count'],
                'claimsRate': target['indemnified'] / target['count'],
                'lossRatio': target['lossRatio']
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
            reduced_rows_complete = map(complete_row, reduced_rows)

            with self.output().open('w') as f:
                writer = csv.DictWriter(f, fieldnames=['year', 'count', 'claimsRate', 'lossRatio'])
                writer.writeheader()
                writer.writerows(reduced_rows_complete)
