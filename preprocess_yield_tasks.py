"""Tasks which preprocess yield estimations from SCYM.

License:
    BSD
"""
import csv
import itertools
import os

import luigi
import shapely
import toolz.itertoolz

import file_util
import const
import cluster_tasks


def process_single(source_filename, access_key='', access_secret=''):
    """Summarize a single yield geotiff.

    Summarize a single yield geotiff within a self-contained function which can be run in
    distributed computing. This has its own inputs and can be exported to other machines.

    Args:
        source_filename: The filename of the input geotiff.
        access_key: The AWS access key to use to access the geotiff if remote. If empty string, will
            treat source_filename as local. Defaults to empty string ('').
        access_secret: The AWS access secret to use to access the geotiff if remote. If empty
            string, will treat source_filename as local. Defaults to empty string ('').

    Returns:
        GeohashYieldSummary
    """
    import os

    import geotiff
    import geolib.geohash
    import libgeohash
    import numpy
    import scipy.stats

    import file_util
    import const
    import data_struct

    def get_temporary_file(source_filename):
        return file_util.save_file_tmp(
            const.BUCKET_OR_DIR,
            source_filename,
            access_key,
            access_secret
        )

    def remove_temporary_file(temp_file_path):
        file_util.remove_temp_file(
            temp_file_path,
            access_key,
            access_secret
        )

    def run(source_filename):
        full_path = get_temporary_file(source_filename)

        assert os.path.isfile(full_path)

        target_geotiff = geotiff.GeoTiff(full_path)

        bounding_box = target_geotiff.tif_bBox_converted

        polygon = shapely.Polygon([
            [bounding_box[0][0], bounding_box[0][1]],
            [bounding_box[0][0], bounding_box[1][1]],
            [bounding_box[1][0], bounding_box[1][1]],
            [bounding_box[1][0], bounding_box[0][1]]
        ])
        all_geohashes = libgeohash.polygon_to_geohash(polygon, precision=4)

        match = const.YEAR_REGEX.match(source_filename)
        year = int(match.group(1))

        summaries = map(
            lambda x: get_geohash_summary(x, year, target_geotiff),
            all_geohashes
        )
        summaries_valid = filter(lambda x: x.get_count() > 0, summaries)
        summaries_realized = list(summaries_valid)

        remove_temporary_file(full_path)

        return summaries_realized

    def get_geohash_summary(geohash, year, geotiff):
        bounds_reverse = geolib.geohash.bounds(geohash)

        bounds = [
            [bounds_reverse[0][1], bounds_reverse[0][0]],
            [bounds_reverse[1][1], bounds_reverse[1][0]]
        ]

        raw_data = geotiff.read_box(bounds)
        values = numpy.extract(raw_data > 0, raw_data)

        count = values.shape[0]
        if count > 0:
            mean = numpy.mean(values)
            std = numpy.std(values)
        else:
            mean = 0
            std = 0

        skew = scipy.stats.skew(values)
        kurtosis = scipy.stats.kurtosis(values)

        return data_struct.GeohashYieldSummary(
            year,
            geohash,
            mean,
            std,
            count,
            skew,
            kurtosis
        )

    return run(source_filename)


class GetYieldGeotiffsTask(luigi.Task):
    """Task to build task information for actual preprocessing.

    Task to build task information for actual preprocessing, enumerating the geohash / year pairs
    and the specific geotiffs to be processed. This information is a list of geotiff filenames.
    """

    def output(self):
        """Get location where the task information should be written.

        Returns:
            LocalTarget at which the task information (list of geotiff filenames) should be written.
        """
        return luigi.LocalTarget(const.get_file_location('yield_tasks.txt'))

    def run(self):
        """Determine the specific tasks to be completed for later preprocessing."""
        candidate_files = self._get_file_listing()
        matching_files = filter(
            lambda x: const.YEAR_REGEX.match(str(x)) is not None,
            candidate_files
        )
        matching_files_no_aux = filter(
            lambda x: '.aux.xml' not in x,
            matching_files
        )

        with self.output().open('w') as f:
            output_str = '\n'.join(matching_files_no_aux)
            f.write(output_str)

    def _get_file_listing(self):
        """Get the list of geotiff files.

        Returns:
            List of files as strings.
        """
        access_key = os.environ.get('AWS_ACCESS_KEY', '')
        access_secret = os.environ.get('AWS_ACCESS_SECRET', '')
        bucket_name = const.BUCKET_OR_DIR
        return file_util.get_bucket_files(bucket_name, access_key, access_secret)


class PreprocessYieldGeotiffsTask(luigi.Task):
    """Preprocess yield information by geohash."""

    def requires(self):
        """Return list of tasks required to run yield preprocessing.

        Returns:
            StartClusterTask and GetYieldGeotiffsTask
        """
        return {
            'cluster': cluster_tasks.StartClusterTask(),
            'tasks': GetYieldGeotiffsTask()
        }

    def output(self):
        """Get the location where the task information should be written.

        Returns:
            LocalTarget where the preprocessed data should be written.
        """
        return luigi.LocalTarget(const.get_file_location('yield.csv'))

    def run(self):
        """Preprocess yield information."""
        input_files = self._get_tasks()

        input_records_nest = self._run_tasks(input_files)
        input_records = itertools.chain(*input_records_nest)

        combined_records_dict = toolz.itertoolz.reduceby(
            lambda x: x.get_key(),
            lambda a, b: a.combine(b),
            input_records
        )
        combined_records_values = combined_records_dict.values()
        output_dicts = map(lambda x: x.to_dict(), combined_records_values)

        with self.output().open('w') as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=const.GEOHASH_YIELD_COLS
            )
            writer.writeheader()
            writer.writerows(output_dicts)

    def _get_tasks(self):
        """Get information about preprocessing tasks.

        Returns:
            List of files to be preprocessed.
        """
        with self.input()['tasks'].open('r') as f:
            all_lines = map(lambda x: x.strip(), f)
            non_empty_lines = filter(lambda x: x != '', all_lines)
            ret_lines = list(non_empty_lines)

        return ret_lines

    def _run_tasks(self, tasks):
        """Preprocess a series of geotiffs.

        Args:
            tasks: The list of geotiff filenames to preprocess.

        Returns:
            List of geotiff filenames.
        """
        cluster = cluster_tasks.get_cluster()
        cluster.adapt(minimum=10, maximum=50)
        client = cluster.get_client()

        access_key = os.environ.get('AWS_ACCESS_KEY', '')
        access_secret = os.environ.get('AWS_ACCESS_SECRET', '')

        futures = client.map(
            lambda x: process_single(x, access_key, access_secret),
            tasks
        )

        return client.gather(futures)


class GetTargetGeohashesTask(luigi.Task):
    """Get the list of geohashes to be included in analysis."""

    def requires(self):
        """Get the task which determines the geotiffs to preprocess.

        Returns:
            PreprocessYieldGeotiffsTask
        """
        return PreprocessYieldGeotiffsTask()

    def run(self):
        """Write out a list of geohashes to be included in analysis as a CSV file."""
        with self.input().open() as f:
            reader = csv.DictReader(f)
            geohashes = set(map(lambda x: x['geohash'], reader))

        with self.output().open('w') as f:
            for geohash in geohashes:
                f.write(geohash)
                f.write('\n')

    def output(self):
        """Indicate where the list of geohashes should be written.

        Returns:
            LocalTarget at which the CSV file should be written.
        """
        return luigi.LocalTarget(const.get_file_location('geohashes.txt'))
