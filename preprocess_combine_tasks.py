"""Tasks to combine preprocessed datasets together.

License:
    BSD
"""
import csv
import itertools

import luigi

import const
import parse_util
import preprocess_climate_tasks
import preprocess_yield_tasks


class GeohashCollectionBuilderBase:
    """Template builder to create yearly summaries for a geohash that include all variables."""

    def __init__(self, geohash):
        """Create a new builder for a geohash.

        Args:
            geohash: The geohash for which this builder will construct yearly summaries.
        """
        self._geohash = geohash
        self._yield_means = []
        self._yield_stds = []
        self._yield_counts = []
        self._years = {}

    def add_year(self, year, yield_mean, yield_std, yield_observations):
        """Start building a new summary for a new year.

        Args:
            year: The year for which a new summary should be started.
            yield_mean: The mean value for yield to use for this summary.
            yield_std: The standard deviation value for yield to use for this summary.
            yield_observations: The sample size / number of observations of yield for this year.
        """
        if self._get_has_any_missing([yield_mean, yield_std, yield_observations]):
            return

        if year in self._years:
            return

        self._yield_means.append(yield_mean)
        self._yield_stds.append(yield_std)
        self._years[year] = TrainingInstanceBuilder(
            year,
            yield_mean,
            yield_std,
            yield_observations
        )

    def add_climate_value(self, year, month, var, mean, std, min_val, max_val, count):
        """Report a value for a climate variable like chirps.

        Args:
            year: The year for which the value is reported.
            month: The month for which the value is reported.
            var: The name of the climate variable like chirps.
            mean: The mean value of the climate variable reported for this month / year.
            std: The standard deviation of the climate variable reported for this month / year.
            min_val: The minimum value observed for the climate variable for this month / year.
            max_val: The maximum value observed for the climate variable for this month / year.
            count: The number of observations / sample size for this climate variable.
        """
        if year not in self._years:
            return

        year_builder = self._years[year]
        year_builder.add_climate_value(month, var, mean, std, min_val, max_val, count)

    def to_dicts(self):
        """Generate yearly summaries for this geohash for all climate variables reported.

        Returns:
            List of primitives-only dictionaries.
        """
        inner_dicts = map(lambda x: x.to_dict(), self._years.values())

        total_count = sum(self._yield_counts)

        def get_weighted_average(target):
            if total_count == 0:
                return None

            paired = zip(target, self._yield_counts)
            product = map(lambda x: x[0] * x[1], paired)
            product_sum = sum(product)
            return product_sum / total_count

        baseline_yield_mean = get_weighted_average(self._yield_means)
        baseline_yield_std = get_weighted_average(self._yield_stds)

        def add_baselines(target):
            """Report the overall average for the geohash across the series as baseline.

            Some statistics or modeling require a "baseline" value which is simpley the overall
            average of average yield and the overall average of yield std for all years for the
            geohash. This adds those values to output dictionaries.

            Args:
                target: The output record to augment.

            Returns:
                The input target with baseline.
            """
            target['geohash'] = self._geohash
            target['baselineYieldMean'] = baseline_yield_mean
            target['baselineYieldStd'] = baseline_yield_std
            return target

        finished_dicts = map(add_baselines, inner_dicts)
        return finished_dicts

    def _add_builder(self, year, builder):
        self._years[year] = builder

    def _has_year(self, year):
        return year in self._years

    def _add_mean_std(self, mean, std, count):
        if self._get_has_any_missing([mean, std, count]):
            return
        
        self._yield_means.append(mean)
        self._yield_stds.append(std)
        self._yield_counts.append(count)
    
    def _get_has_any_missing(self, required_fields):
        missing_fields = filter(lambda x: x is None, required_fields)
        num_missing_fields = sum(map(lambda x: 1, missing_fields))
        has_missing_fields = num_missing_fields > 0
        return has_missing_fields


class GeohashCollectionBuilder(GeohashCollectionBuilderBase):
    """Builder to create yearly summaries for a geohash that include all variables."""

    def add_year(self, year, yield_mean, yield_std, yield_observations):
        """Start building a new summary for a new year.

        Args:
            year: The year for which a new summary should be started.
            yield_mean: The mean value for yield to use for this summary.
            yield_std: The standard deviation value for yield to use for this summary.
            yield_observations: The sample size / number of observations of yield for this year.
        """
        if self._get_has_any_missing([yield_mean, yield_std, yield_observations]):
            return

        if self._has_year(year):
            return

        self._add_mean_std(yield_mean, yield_std, yield_observations)

        builder = TrainingInstanceBuilder(
            year,
            yield_mean,
            yield_std,
            yield_observations
        )
        self._add_builder(year, builder)


class GeohashCollectionBetaBuilder(GeohashCollectionBuilderBase):
    """Builder to create yearly summaries for a geohash that include all variables.

    Builder to create yearly summaries for a geohash that include all variables with a beta
    distribution.
    """

    def add_year(self, year, yield_mean, yield_std, yield_a, yield_b, yield_loc, yield_scale,
        yield_observations):
        """Start building a new summary for a new year.

        Args:
            year: The year for which a new summary should be started.
            yield_mean: The mean value for yield to use for this summary.
            yield_std: The standard deviation value for yield to use for this summary.
            yield_observations: The sample size / number of observations of yield for this year.
            yield_a: Parameter a for the beta distribution.
            yield_b: Parameter b for the beta distribution.
            yield_loc: Center / location for the beta distribution.
            yield_scale: Scale for the beta distribution.
        """
        required_fields = [
            yield_mean,
            yield_std,
            yield_observations,
            yield_a,
            yield_b,
            yield_loc,
            yield_scale
        ]

        if self._get_has_any_missing(required_fields):
            return

        if self._has_year(year):
            return

        self._add_mean_std(yield_mean, yield_std, yield_observations)

        builder = TrainingInstanceBetaBuilder(
            year,
            yield_a,
            yield_b,
            yield_loc,
            yield_scale,
            yield_observations
        )
        self._add_builder(year, builder)


class TrainingInstanceBuilderBase:
    """Builder to generate model training a single year summary for a single geohash."""

    def __init__(self):
        """Create a new builder with empty climate info."""
        self._climate_means = {}
        self._climate_stds = {}
        self._climate_mins = {}
        self._climate_maxes = {}
        self._climate_counts = {}
        self._total_climate_counts = 0
        self._keys = set()

    def add_climate_value(self, month, var, mean, std, min_val, max_val, count):
        """Indicate the value of a climate variable observed within this year.

        Args:
            month: The month for which the value is reported.
            var: The name of the climate variable like chirps.
            mean: The mean value of the climate variable reported for this month / year.
            std: The standard deviation of the climate variable reported for this month / year.
            min_val: The minimum value observed for the climate variable for this month / year.
            max_val: The maximum value observed for the climate variable for this month / year.
            count: The number of observations / sample size for this climate variable.
        """
        var_rename = const.CLIMATE_VARIABLE_TO_ATTR[var]
        key = '%s/%d' % (var_rename, month)

        assert key not in self._keys

        self._keys.add(key)
        self._climate_means[key] = mean
        self._climate_stds[key] = std
        self._climate_mins[key] = min_val
        self._climate_maxes[key] = max_val
        self._climate_counts[key] = count
        self._total_climate_counts += 1

    def to_dict(self):
        """Generate a single output record describing all variables for this year.

        Returns:
            Primitives-only dictionary representing this geohash for this year.
        """
        output_dict = {}

        for key in self._keys:
            (var_rename, month) = key.split('/')
            get_output_key = lambda x: ''.join([var_rename, x, month])

            output_dict[get_output_key('Mean')] = self._climate_means[key]
            output_dict[get_output_key('Std')] = self._climate_stds[key]
            output_dict[get_output_key('Min')] = self._climate_mins[key]
            output_dict[get_output_key('Max')] = self._climate_maxes[key]
            output_dict[get_output_key('Count')] = self._climate_counts[key]

        self._finalize_output(output_dict)

        return output_dict

    def _finalize_output(self, output_dict):
        raise NotImplementedError('Use implementor.')


class TrainingInstanceBuilder(TrainingInstanceBuilderBase):
    """Builder to generate model training a single year summary for a single geohash."""

    def __init__(self, year, yield_mean, yield_std, yield_observations):
        """Create a new builder.

        Args:
            year: The year for which training instances are being generated.
            yield_mean: The average yield for the year.
            yield_std: The standard deviation of yield for the year.
            yield_observations: Sample size / observation count for yield.
        """
        super().__init__()
        self._year = year
        self._yield_mean = yield_mean
        self._yield_std = yield_std
        self._yield_observations = yield_observations

    def _finalize_output(self, output_dict):
        output_dict['year'] = self._year
        output_dict['climateCounts'] = self._total_climate_counts
        output_dict['yieldMean'] = self._yield_mean
        output_dict['yieldStd'] = self._yield_std
        output_dict['yieldObservations'] = self._yield_observations
        return output_dict


class TrainingInstanceBetaBuilder(TrainingInstanceBuilderBase):
    """Builder to generate model training a single year summary for a single geohash."""

    def __init__(self, year, yield_a, yield_b, yield_loc, yield_scale, yield_observations):
        """Create a new builder.

        Args:
            year: The year for which training instances are being generated.
            yield_a: Parameter a for the beta distribution.
            yield_b: Parameter b for the beta distribution.
            yield_loc: Center / location for the beta distribution.
            yield_scale: Scale for the beta distribution.
            yield_observations: Sample size / observation count for yield.
        """
        super().__init__()
        self._year = year
        self._yield_a = yield_a
        self._yield_b = yield_b
        self._yield_loc = yield_loc
        self._yield_scale = yield_scale
        self._yield_observations = yield_observations

    def _finalize_output(self, output_dict):
        output_dict['year'] = self._year
        output_dict['climateCounts'] = self._total_climate_counts
        output_dict['yieldA'] = self._yield_a
        output_dict['yieldB'] = self._yield_b
        output_dict['yieldLoc'] = self._yield_loc
        output_dict['yieldScale'] = self._yield_scale
        output_dict['yieldObservations'] = self._yield_observations
        return output_dict


class CombineHistoricPreprocessTemplateTask(luigi.Task):
    """Template to combine geohash summaries (yield and climate) for a historic series."""

    def requires(self):
        """Indicate that preprocessed climate and yields data are required.

        Returns:
            PreprocessClimateGeotiffsTask and PreprocessYieldGeotiffsTask
        """
        return {
            'climate': preprocess_climate_tasks.PreprocessClimateGeotiffsTask(
                dataset_name='historic',
                conditions=['observations'],
                years=const.YEARS
            ),
            'yield': self._get_yield_task()
        }

    def output(self):
        """Indicate where the combined summaries should be written.

        Returns:
            LocalTarget at which these combined summaries should be written.
        """
        return luigi.LocalTarget(const.get_file_location(self._get_filename()))

    def run(self):
        """Generate combined summaries."""
        geohash_builders = {}

        self._process_yields(geohash_builders)

        with self.input()['climate'].open('r') as f:
            rows = csv.DictReader(f)

            for row in rows:
                geohash = str(row['geohash'])
                year = int(row['year'])
                month = int(row['month'])
                var = str(row['var'])
                mean = float(row['mean'])
                std = float(row['std'])
                min_val = float(row['min'])
                max_val = float(row['max'])
                count = float(row['count'])

                geohash_builder = geohash_builders[geohash]
                geohash_builder.add_climate_value(
                    year,
                    month,
                    var,
                    mean,
                    std,
                    min_val,
                    max_val,
                    count
                )

        builders_flat = geohash_builders.values()
        dicts_nested = map(lambda x: x.to_dicts(), builders_flat)
        dicts = itertools.chain(*dicts_nested)

        with self.output().open('w') as f:
            writer = csv.DictWriter(f, fieldnames=self._get_output_attrs())
            writer.writeheader()
            writer.writerows(dicts)

    def _get_output_attrs(self):
        return const.TRAINING_FRAME_ATTRS

    def _process_yields(self, geohash_builders):
        raise NotImplementedError('Use implementor.')

    def _get_yield_task(self):
        raise NotImplementedError('Use implementor.')

    def _get_filename(self):
        raise NotImplementedError('Use implementor.')


class CombineHistoricPreprocessTask(CombineHistoricPreprocessTemplateTask):
    """Combine geohash summaries (yield and climate) for a historic series."""

    def _get_output_attrs(self):
        return const.TRAINING_FRAME_ATTRS

    def _process_yields(self, geohash_builders):
        with self.input()['yield'].open('r') as f:
            rows = csv.DictReader(f)

            for row in rows:
                year = int(row['year'])
                geohash = str(row['geohash'])
                mean = float(row['mean'])
                std = float(row['std'])
                count = float(row['count'])

                if geohash not in geohash_builders:
                    geohash_builders[geohash] = GeohashCollectionBuilder(geohash)

                geohash_builder = geohash_builders[geohash]
                geohash_builder.add_year(year, mean, std, count)

        return geohash_builders

    def _get_yield_task(self):
        return preprocess_yield_tasks.PreprocessYieldGeotiffsTask()

    def _get_filename(self):
        return 'training_frame.csv'


class CombineHistoricPreprocessBetaTask(CombineHistoricPreprocessTemplateTask):
    """Combine geohash summaries (yield and climate) for a historic series with beta dist."""

    def _get_output_attrs(self):
        return const.TRAINING_FRAME_BETA_ATTRS

    def _process_yields(self, geohash_builders):
        with self.input()['yield'].open('r') as f:
            rows = csv.DictReader(f)

            for row in rows:
                year = parse_util.try_int(row['year'])
                geohash = str(row['geohash'])
                mean = parse_util.try_float(row['mean'])
                std = parse_util.try_float(row['std'])
                yield_a = parse_util.try_float(row['a'])
                yield_b = parse_util.try_float(row['b'])
                yield_loc = parse_util.try_float(row['loc'])
                yield_scale = parse_util.try_float(row['scale'])
                count = parse_util.try_float(row['count'])

                required_fields = [
                    year,
                    geohash,
                    mean,
                    std,
                    yield_a,
                    yield_b,
                    yield_loc,
                    yield_scale,
                    count
                ]
                missing_fields = filter(lambda x: x is None, required_fields)
                num_missing_fields = sum(map(lambda x: 1, missing_fields))

                if geohash not in geohash_builders:
                    geohash_builders[geohash] = GeohashCollectionBetaBuilder(geohash)

                geohash_builder = geohash_builders[geohash]
                geohash_builder.add_year(
                    year,
                    mean,
                    std,
                    count,
                    yield_a,
                    yield_b,
                    yield_loc,
                    yield_scale
                )

        return geohash_builders

    def _get_yield_task(self):
        return preprocess_yield_tasks.PreprocessYieldGeotiffsBetaTask()

    def _get_filename(self):
        return 'training_frame_beta.csv'


class ReformatFuturePreprocessTemplateTask(luigi.Task):
    """Template for task creating a model-compatible frame in which future yields can be predicted.

    Create a model-compatible frame containing climate projections in a format in which future
    yields can be predicted.
    """

    def run(self):
        """Run the reformatting."""
        geohash_builders = {}

        with self.input()['climate'].open('r') as f:
            rows = csv.DictReader(f)

            for row in rows:
                geohash = str(row['geohash'])
                year = int(row['year'])
                month = int(row['month'])
                var = str(row['var'])
                mean = float(row['mean'])
                std = float(row['std'])
                min_val = float(row['min'])
                max_val = float(row['max'])
                count = float(row['count'])

                geohash_builder = self._get_geohash_builder(geohash, year, geohash_builders)

                geohash_builder.add_climate_value(
                    year,
                    month,
                    var,
                    mean,
                    std,
                    min_val,
                    max_val,
                    count
                )

        builders_flat = geohash_builders.values()
        dicts_nested = map(lambda x: x.to_dicts(), builders_flat)
        dicts = itertools.chain(*dicts_nested)

        with self.output().open('w') as f:
            writer = csv.DictWriter(f, fieldnames=self._get_output_attrs())
            writer.writeheader()
            writer.writerows(dicts)

    def _get_output_attrs(self):
        raise NotImplementedError('Use implementor.')

    def _get_geohash_builder(self, geohash, year):
        raise NotImplementedError('Use implementor.')


class ReformatFuturePreprocessTask(ReformatFuturePreprocessTemplateTask):
    """Template for task creating a model-compatible frame in which future yields can be predicted.

    Create a model-compatible frame containing climate projections in a format in which future
    yields can be predicted.
    """

    condition = luigi.Parameter()

    def requires(self):
        """Indicate that climate data are required.

        Returns:
            PreprocessClimateGeotiffsTask
        """
        return {
            'climate': preprocess_climate_tasks.PreprocessClimateGeotiffsTask(
                dataset_name=self.condition,
                conditions=[self.condition],
                years=const.FUTURE_REF_YEARS
            )
        }

    def output(self):
        """Indicate the location at which the reformatted data frame should be written.

        Returns:
            LocalTarget at which the reformatted data should be written.
        """
        return luigi.LocalTarget(const.get_file_location('%s_frame.csv' % self.condition))

    def _get_output_attrs(self):
        return const.TRAINING_FRAME_ATTRS

    def _get_geohash_builder(self, geohash, year, geohash_builders):
        if geohash not in geohash_builders:
            geohash_builders[geohash] = GeohashCollectionBuilder(geohash)

        geohash_builder = geohash_builders[geohash]
        geohash_builder.add_year(year, -1, -1, -1)

        return geohash_builder


class ReformatFuturePreprocessBetaTask(ReformatFuturePreprocessTemplateTask):
    """Template for task creating a model-compatible frame in which future yields can be predicted.

    Create a model-compatible frame containing climate projections in a format in which future
    yields can be predicted using a beta distribution.
    """

    condition = luigi.Parameter()

    def requires(self):
        """Indicate that climate data are required.

        Returns:
            PreprocessClimateGeotiffsTask
        """
        return {
            'climate': preprocess_climate_tasks.PreprocessClimateGeotiffsTask(
                dataset_name=self.condition,
                conditions=[self.condition],
                years=const.FUTURE_REF_YEARS
            )
        }

    def output(self):
        """Indicate the location at which the reformatted data frame should be written.

        Returns:
            LocalTarget at which the reformatted data should be written.
        """
        return luigi.LocalTarget(const.get_file_location('%s_frame_beta.csv' % self.condition))

    def _get_output_attrs(self):
        return const.TRAINING_FRAME_BETA_ATTRS

    def _get_geohash_builder(self, geohash, year, geohash_builders):
        if geohash not in geohash_builders:
            geohash_builders[geohash] = GeohashCollectionBetaBuilder(geohash)

        geohash_builder = geohash_builders[geohash]
        geohash_builder.add_year(year, -1, -1, -1, -1, -1, -1, -1)

        return geohash_builder
