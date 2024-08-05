import csv
import itertools
import os

import luigi
import shapely
import toolz.itertoolz

import file_util
import const
import cluster_tasks


def process_single(remote_filename, access_key, access_secret):
    import os

    import geotiff
    import geolib.geohash
    import libgeohash
    import numpy
    import scipy.stats

    import file_util
    import const
    import data_struct

    def get_temporary_file(remote_filename):
        return file_util.save_file_tmp(
            const.BUCKET_OR_DIR,
            remote_filename,
            access_key,
            access_secret
        )

    def remove_temporary_file(temp_file_path):
        file_util.remove_temp_file(
            temp_file_path,
            access_key,
            access_secret
        )

    def run(remote_filename):
        full_path = get_temporary_file(remote_filename)

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

        match = const.YEAR_REGEX.match(remote_filename)
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

    return run(remote_filename)


class GetYieldGeotiffsTask(luigi.Task):

    def output(self):
        return luigi.LocalTarget(const.get_file_location('yield_tasks.txt'))

    def run(self):
        candidate_files = self._get_remote_file_listing()
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

    def _get_remote_file_listing(self):
        access_key = os.environ.get('AWS_ACCESS_KEY', '')
        access_secret = os.environ.get('AWS_ACCESS_SECRET', '')
        bucket_name = const.BUCKET_OR_DIR
        return file_util.get_bucket_files(bucket_name, access_key, access_secret)


class PreprocessYieldGeotiffsTask(luigi.Task):

    def requires(self):
        return {
            'cluster': cluster_tasks.StartClusterTask(),
            'tasks': GetYieldGeotiffsTask()
        }

    def output(self):
        return luigi.LocalTarget(const.get_file_location('yield.csv'))

    def run(self):
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
        with self.input()['tasks'].open('r') as f:
            all_lines = map(lambda x: x.strip(), f)
            non_empty_lines = filter(lambda x: x != '', all_lines)
            ret_lines = list(non_empty_lines)

        return ret_lines

    def _run_tasks(self, tasks):
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

    def requires(self):
        return PreprocessYieldGeotiffsTask()

    def run(self):
        with self.input().open() as f:
            reader = csv.DictReader(f)
            geohashes = set(map(lambda x: x['geohash'], reader))

        with self.output().open('w') as f:
            for geohash in geohashes:
                f.write(geohash)
                f.write('\n')

    def output(self):
        return luigi.LocalTarget(const.get_file_location('geohashes.txt'))
