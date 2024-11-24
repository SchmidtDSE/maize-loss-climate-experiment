"""Tasks to summarize climate variable geotiffs (CHC-CMIP6) by geohash.

License:
    BSD
"""
import csv
import datetime
import itertools
import os

import dask.bag
import luigi

import cluster_tasks
import const
import data_struct
import preprocess_yield_tasks


def get_input_tiffs(sample_day_gap, years, variables, conditions):
    """Get the list of tiffs within a series.

    Args:
        sample_day_gap: How many dasy between each sample or 1 for all days.
        years: List of integer years to include the series.
        variables: Variables like chirps to include in the series.
        conditions: Conditions like 2050_SSP245 to include in the series.

    Returns:
        List of InputTiff
    """
    def get_date_in_year(year):
        date = datetime.date(year, 1, 1)
        dates = []
        while date.year == year:
            dates.append(date)
            date = date + datetime.timedelta(days=sample_day_gap)
        return dates

    dates_nested = map(get_date_in_year, years)
    dates = itertools.chain(*dates_nested)

    combinations = itertools.product(variables, conditions, dates)

    input_tiffs = map(lambda x: data_struct.InputTiff(x[0], x[1], x[2]), combinations)

    return input_tiffs


def get_daily_geohash(source, tiff_info, geohashes, access_key='', access_secret=''):
    """Generate summaries for all geohashes in a geotiff.

    Self-contained function which generate summaries for all geohashes in a geotiff. This can be
    executed within distribution and has its own import statements.

    Args:
        source: Location such as bucket name or local directory where the geotiff can be found.
            Assumed to be a bucket name if both access_key and access_secret at provided. Otherwise,
            assumed to be a local path.
        tiff_info: Information about the geotiff to process.
        geohashes: Geohashes to summarize. Others not included will be ignored.
        access_key: Optional AWS access key or empty string ('') if source is local. Defaults to
            empty string indicating that source is local.
        access_secret: Optional AWS access secret or empty string ('') if source is local. Defaults
            to empty string indicating that source is local.

    Returns:
        One GeohashClimateSummary per geohash.
    """
    import geolib.geohash
    import geotiff
    import numpy
    import scipy.stats

    import file_util
    import data_struct
    import distribution_struct

    tiff_filename = tiff_info.get_filename()

    temp_file_path = file_util.save_file_tmp(
        source,
        tiff_filename,
        access_key,
        access_secret
    )

    try:
        tiff = geotiff.GeoTiff(temp_file_path)
    except:
        print('Failed to get for ' + tiff_filename)

        try:
            file_util.remove_temp_file(
                temp_file_path,
                access_key,
                access_secret
            )
        except:
            print('Could not remove ' + temp_file_path)

        return []

    tiff_data = tiff.read()

    def get_geohash_dist(geohash):
        bounds_reverse = geolib.geohash.bounds(geohash)

        bounds = [
            [bounds_reverse[0][1], bounds_reverse[0][0]],
            [bounds_reverse[1][1], bounds_reverse[1][0]]
        ]

        indicies = tiff.get_int_box(bounds, outer_points=1)

        start_x = indicies[0][0]
        end_x = indicies[1][0]
        start_y = indicies[0][1]
        end_y = indicies[1][1]

        raw_data_all = tiff_data[start_y:end_y, start_x:end_x]
        raw_data = numpy.extract(raw_data_all >= 0, raw_data_all)

        dist_count = raw_data.shape[0]
        if dist_count == 0:
            msg_vals = (geohash, tiff_filename)
            print('Encountered no data on %s for %s' % msg_vals)
            return None

        dist_mean = numpy.mean(raw_data)
        dist_std = numpy.std(raw_data)
        dist_min = numpy.min(raw_data)
        dist_max = numpy.max(raw_data)
        dist_skew = scipy.stats.skew(raw_data)
        dist_kurtosis = scipy.stats.kurtosis(raw_data)

        return distribution_struct.Distribution(
            dist_mean,
            dist_std,
            dist_count,
            dist_min,
            dist_max,
            dist_skew,
            dist_kurtosis
        )

    def make_geohash_summary(geohash, distribution):
        return data_struct.GeohashClimateSummary(
            geohash,
            tiff_info.get_date().year,
            tiff_info.get_date().month,
            tiff_info.get_variable(),
            tiff_info.get_condition(),
            distribution.get_mean(),
            distribution.get_std(),
            distribution.get_min(),
            distribution.get_max(),
            distribution.get_count(),
            distribution.get_skew(),
            distribution.get_kurtosis(),
            tiff_info.get_date().day
        )

    distributions_all = map(lambda x: (x, get_geohash_dist(x)), list(geohashes))
    distributions = filter(lambda x: x[1] is not None, distributions_all)
    summaries = map(lambda x: make_geohash_summary(x[0], x[1]), distributions)
    summaries_realized = list(summaries)

    file_util.remove_temp_file(
        temp_file_path,
        access_key,
        access_secret
    )

    return summaries_realized


def get_geohash_summary_key(geohash_summary):
    """Convienence function to get the unique key for a geohash summary.

    Args:
        geohash_summary: The summary object for which the key is desired.

    Returns:
        String uniquely identifying a geohash for a variable and condition at a point in time.
    """
    return geohash_summary.get_key()


def combine_summaries(first_record, second_record):
    """Convienence function to pool samples for a geohash summary.

    Args:
        first_record: The first summary whose sample is to be pooled.
        second_record: The second summary whose sample is to be pooled.

    Returns:
        Summary after sample pooling.
    """
    return first_record.combine(second_record)


def get_without_day(geohash_summary):
    """Convienence function to get a copy of a geohash summary without day specified.

    Convienence function to get a copy of a geohash summary without day specified, indicating that
    the sample is for a whole month.

    Args:
        geohash_summary: Summary to get without day.

    Returns:
        Copy of summary with day removed.
    """
    geohash_summary_no_day = geohash_summary.get_without_day()
    return geohash_summary_no_day


def confirm_and_standardize_output(target):
    """Standardize and confirm expected fields and types for a climate geohash summary record.

    Args:
        target: The summary as primitives-only dictionary to confirm and standardize.

    Returns:
        Primitives-only dictionary with standardized types and field names.
    """
    return {
        'geohash': str(target['geohash']),
        'year': int(target['year']),
        'month': int(target['month']),
        'var': str(target['var']),
        'condition': str(target['condition']),
        'mean': float(target['mean']),
        'std': float(target['std']),
        'min': float(target['min']),
        'max': float(target['max']),
        'count': int(target['count']),
        'skew': float(target['skew']),
        'kurtosis': float(target['kurtosis'])
    }


def run_job(sample_day_gap, source, years, variables, geohashes, conditions, cluster, access_key='',
    access_secret=''):
    """Preprocess a collection of geotiffs.

    Ask the cluster or other distributed computing client to execute a climate geohash preprocessing
    job which includes a collection or batch of geotiffs.

    Args:
        sample_day_gap: How many dasy between each sample or 1 for all days.
        source: Location such as bucket name or local directory where the geotiff can be found.
            Assumed to be a bucket name if both access_key and access_secret at provided. Otherwise,
            assumed to be a local path.
        years: List of integer years to include the series.
        variables: List of variables like chirps to include in the series.
        geohashes: List of geohashes to summarize. Geohashes not listed will be ignored.
        conditions: List of conditions like 2050_SSP245 to include in the series.
        cluster: Client to execute distributed tasks.
        access_key: AWS access key to use if source is remote or empty string ('') if source is
            local. Defaults to empty string meaning that AWS is not engaged.
        access_secret: AWS access secret to use if source is remote or empty string ('') if source
            is local. Defaults to empty string meaning that AWS is not engaged.

    Returns:
        Collection of geohash summaries as primitives-only dictionaries.
    """
    client = cluster.get_client()

    input_tiffs = get_input_tiffs(sample_day_gap, years, variables, conditions)

    geohash_summaries_nested_future = client.map(
        lambda x: get_daily_geohash(source, x, geohashes, access_key, access_secret),
        list(input_tiffs)
    )
    geohash_summaries_nested = dask.bag.from_sequence(geohash_summaries_nested_future)
    geohash_summaries = geohash_summaries_nested.flatten()
    geohash_summaries_no_day = geohash_summaries.map(get_without_day)
    combined_geohashes_keyed = geohash_summaries_no_day.foldby(
        get_geohash_summary_key,
        combine_summaries
    )
    combined_geohashes = combined_geohashes_keyed.map(lambda x: x[1])
    combined_geohashes_dicts = combined_geohashes.map(lambda x: x.to_dict())
    output_dicts = combined_geohashes_dicts.map(confirm_and_standardize_output)

    output_dicts_realized = output_dicts.compute()

    return output_dicts_realized


class PreprocessClimateGeotiffTask(luigi.Task):
    """Task to preprocess a single geotiff collection."""

    dataset_name = luigi.Parameter()
    conditions = luigi.Parameter()
    year = luigi.Parameter()

    def requires(self):
        """Indicate that the cluster and geohash list are needed.

        Returns:
            Requirement for StartClusterTask and GetTargetGeohashesTask.
        """
        return {
            'cluster': cluster_tasks.StartClusterTask(),
            'geohashes': preprocess_yield_tasks.GetTargetGeohashesTask()
        }

    def output(self):
        """Indicate where the climate summaries should be written.

        Returns:
            LocalTarget at which the summaries should be written.
        """
        filename = 'climate_%s_%d.csv' % (self.dataset_name, self.year)
        return luigi.LocalTarget(const.get_file_location(filename))

    def run(self):
        """Run request for a single collection of geotiffs."""
        cluster = cluster_tasks.get_cluster()
        cluster.adapt(minimum=10, maximum=450)
        geohashes_set = self._get_geohashes()
        tasks = self._get_tasks()

        with self.output().open('w') as f:
            writer = csv.DictWriter(f, fieldnames=const.EXPECTED_CLIMATE_COLS)
            writer.writeheader()

            for task in tasks:
                results = run_job(
                    const.SAMPLE_DAY_GAP,
                    const.BUCKET_OR_DIR,
                    task['years'],
                    task['variables'],
                    geohashes_set,
                    self.conditions,
                    cluster,
                    os.environ.get('AWS_ACCESS_KEY', ''),
                    os.environ.get('AWS_ACCESS_SECRET', '')
                )
                writer.writerows(results)
                f.flush()

    def _get_geohashes(self):
        """Get the list of geohashes for which summaries should be returned.

        Returns:
            Set of strings each representing an individual geohash.
        """
        with self.input()['geohashes'].open('r') as f:
            geohashes = f.readlines()
            geohashes_clean = map(lambda x: x.strip(), geohashes)
            geohashes_allowed = filter(lambda x: x != '', geohashes_clean)
            geohashes_capped = map(
                lambda x: x[:const.GEOHASH_LEN],
                geohashes_allowed
            )
            geohashes_set = set(geohashes_capped)

        return geohashes_set

    def _get_tasks(self):
        """Get tasks for this single collection of geohash summaries.

        Returns:
            List of tasks for a single collectin of geohash summaries.
        """
        years = [self.year]
        variables = const.CLIMATE_VARIABLES

        tasks_all = map(lambda x: {'year': x, 'var': variables}, years)
        tasks_nest = map(
            lambda x: {'years': [x['year']], 'variables': x['var']},
            tasks_all
        )
        tasks = list(tasks_nest)
        return tasks


class PreprocessClimateGeotiffsTask(luigi.Task):
    """Preprocess many collections of geotiffs with climate data."""

    dataset_name = luigi.Parameter()
    conditions = luigi.Parameter()
    years = luigi.Parameter()

    def requires(self):
        """Break the collections up into subtasks.

        Returns:
            List of subtasks represented each as a PreprocessClimateGeotiffTask.
        """
        def make_subtask(year):
            return PreprocessClimateGeotiffTask(
                dataset_name=self.dataset_name,
                conditions=self.conditions,
                year=year
            )
        return [make_subtask(year) for year in self.years]

    def output(self):
        """Indicate where the combined set of geohash summaries should be written.

        Returns:
            LocalTarget at which the summaries should be written.
        """
        return luigi.LocalTarget(
            const.get_file_location('climate_%s.csv' % self.dataset_name)
        )

    def run(self):
        """Run all summarization tasks."""
        with self.output().open('w') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=const.EXPECTED_CLIMATE_COLS)
            writer.writeheader()

            for sub_input in self.input():
                with sub_input.open() as f_in:
                    reader = csv.DictReader(f_in)
                    writer.writerows(reader)
