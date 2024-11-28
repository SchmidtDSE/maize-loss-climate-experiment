"""General data structures used in multiple tasks or task modules.

Shared data structures used in multiple tasks or task modules where note that treatment of yield
may refer to yield deltas but this depends on the pipeline configuration. Note that this file
includes structures specific to this pipeline but distribution_struct provides some more general
mathematical structures. Skew and kurtosis are the values observed under defaults from scipy.stats.

License:
    BSD
"""
import datetime

import numpy
import scipy.stats

import distribution_struct


class GeohashClimateSummary:
    """Object describing a single climate variable for a single geohash.

    Object describing a single climate variable for a single geohash in a single month like
    July 2024. May also represent a single day if day is set.
    """

    def __init__(self, geohash, year, month, var, condition, mean, std, var_min, var_max, count,
        skew, kurtosis, day=None):
        """Create a new climate summary record.

        Args:
            geohash: The string geohash that this summary represents.
            year: The year for which data are reported.
            month: The 1 indexed month for which data are reported (July = 7).
            var: The name of the variable summarized.
            condition: The condition in which this is observed like 2050_SSP245.
            mean: Average of the climate variable seen within the geohash for this year / month.
            std: Std of the climate variable seen within the geohash for this year / month.
            var_min: The minimum value of the variable observed within the geohash for this year /
                month.
            var_max: The maximum value of the variable observed within the geohash for this year /
                month.
            count: The number of observations summarized.
            skew: The skew (how far off from zero) observed in the distribution summarized.
            kurtosis: The kurtosis (width of tails) observed in the distribution summarized.
            day: Optional day for which these data are summarized. If provided, this is the summary
                of the geohash for this year / month / day and not just year / month. Defaults to
                None meaning that the geohash summary is for a month.
        """
        self._geohash = geohash
        self._year = year
        self._month = month
        self._var = var
        self._condition = condition
        self._mean = mean
        self._std = std
        self._var_min = var_min
        self._var_max = var_max
        self._count = count
        self._skew = skew
        self._kurtosis = kurtosis
        self._day = day

    def get_geohash(self):
        """Get the name of the geohash summarized.

        Returns:
            The string geohash that this summary represents.
        """
        return self._geohash

    def get_year(self):
        """Get the year for which this summary is provided.

        Returns:
            The year for which data are reported.
        """
        return self._year

    def get_month(self):
        """Get the month for which this summary is provided.

        Returns:
            The 1 indexed month for which data are reported (July = 7).
        """
        return self._month

    def get_var(self):
        """Get the name of the variable summarized like "chirps".

        Returns:
            The name of the variable summarized as a string.
        """
        return self._var

    def get_condition(self):
        """Get the condition in which these data were observed, either historic or projected.

        Returns:
            The condition in which this is observed like 2050_SSP245 meaning 2050 series and SSP245
            scenario.
        """
        return self._condition

    def get_mean(self):
        """Get the average of the climate variable.

        Returns:
            Average of the climate variable seen within the geohash for this year / month.
        """
        return self._mean

    def get_std(self):
        """Get the standard deviation of the climate variable.

        Returns:
            Std of the climate variable seen within the geohash for this year / month.
        """
        return self._std

    def get_min(self):
        """Get the minimum of the climate variable.

        Returns:
            The minimum value of the variable observed within the geohash for this year / month.
        """
        return self._var_min

    def get_max(self):
        """Get the maximum of the climate variable.

        Returns:
            The maximum value of the variable observed within the geohash for this year / month.
        """
        return self._var_max

    def get_count(self):
        """Get the number of observations summarized.

        Returns:
            The number of observations summarized.
        """
        return self._count

    def get_day(self):
        """Get the day of month summarized if this summary describes a single day or None if month.

        Returns:
            Optional day for which these data are summarized. If provided, this is the summary of
            the geohash for this year / month / day and not just year / month. If None, this means
            that the geohash summary is for a full month and not a specific day.
        """
        return self._day

    def get_skew(self):
        """Get the skew measure of the distribution summarized.

        Returns:
            The skew (how far off from zero) observed in the distribution summarized.
        """
        return self._skew

    def get_kurtosis(self):
        """Get the kurtosis measure of the distribution summarized.

        Returns:
            The kurtosis (width of tails) observed in the distribution summarized.
        """
        return self._kurtosis

    def get_without_day(self):
        """Create a copy of this summary with the day set to None.

        Returns:
            Copy of this summary but with the structure indicating that it summarizes a full month
            instead of a single day.
        """
        return GeohashClimateSummary(
            self._geohash,
            self._year,
            self._month,
            self._var,
            self._condition,
            self._mean,
            self._std,
            self._var_min,
            self._var_max,
            self._count,
            self._skew,
            self._kurtosis
        )

    def get_key(self):
        """Get a string that uniquely identifies the population summarized.

        Returns:
            String which uniquely identifies combination of variable summarized, geohash, date, and
            condition including if this summary is for a full month or just a day.
        """
        components = [
            self.get_geohash(),
            self.get_year(),
            self.get_month(),
            self.get_day(),
            self.get_var(),
            self.get_condition()
        ]
        components_valid = map(lambda x: '' if x is None else x, components)
        components_str = map(lambda x: str(x), components_valid)
        return '\t'.join(components_str)

    def to_dict(self):
        """Convert this structure to a primitives-only dictionary.

        Returns:
            Serialized version.
        """
        return {
            'geohash': self.get_geohash(),
            'year': self.get_year(),
            'month': self.get_month(),
            'var': self.get_var(),
            'condition': self.get_condition(),
            'mean': self.get_mean(),
            'std': self.get_std(),
            'min': self.get_min(),
            'max': self.get_max(),
            'count': self.get_count(),
            'skew': self.get_skew(),
            'kurtosis': self.get_kurtosis(),
            'day': self.get_day()
        }

    def get_distribution(self):
        """Get a structure describing the distribution found within this geohash.

        Returns:
            Distribution object describing the values found within this geohash / date combination.
        """
        return distribution_struct.Distribution(
            self.get_mean(),
            self.get_std(),
            self.get_count(),
            dist_min=self.get_min(),
            dist_max=self.get_max(),
            skew=self.get_skew(),
            kurtosis=self.get_kurtosis()
        )

    def combine(self, other, allow_multiple_shapes=False):
        """Combine the samples of two geohash summaries.

        Args:
            other: The geohash with additional data on this summary's sample.
            allow_multiple_shapes: Flag indicating if different or non-normal distribution shapes
                can be combined.

        Returns:
            New summary after combining samples.
        """
        self_dist = self.get_distribution()
        other_dist = other.get_distribution()
        new_dist = self_dist.combine(other_dist, allow_multiple_shapes=allow_multiple_shapes)

        assert self.get_key() == other.get_key()
        return GeohashClimateSummary(
            self.get_geohash(),
            self.get_year(),
            self.get_month(),
            self.get_var(),
            self.get_condition(),
            new_dist.get_mean(),
            new_dist.get_std(),
            new_dist.get_min(),
            new_dist.get_max(),
            new_dist.get_count(),
            new_dist.get_skew(),
            new_dist.get_kurtosis(),
            self.get_day()
        )


def parse_geohash_climate_summary(target_dict):
    """Parse a climate summary from a primitives only dictionary.

    Args:
        target_dict: The dictionary form which to deserialize.

    Returns:
        Deserialized climate summary.
    """
    return GeohashClimateSummary(
        target_dict['geohash'],
        target_dict['year'],
        target_dict['month'],
        target_dict['var'],
        target_dict['mean'],
        target_dict['std'],
        target_dict['min'],
        target_dict['max'],
        target_dict['count'],
        target_dict['skew'],
        target_dict['kurtosis'],
        target_dict['day']
    )


class GeohashYieldSummary:
    """Summary of distribution of yields found within a geohash for a single year."""

    def __init__(self, year, geohash, mean, std, count, skew, kurtosis):
        """Create a new summary.

        Args:
            year: The year of this summary.
            geohash: The name of geohash summarized.
            mean: The mean of the observed yields.
            std: The standard deviation of the observed yields.
            count: The number of observations / data points summarized.
            skew: The measure of skew of the yields distribution (how far off center).
            kurtosis: The measure of the yields kurtosis (tail thickness).
        """
        self._year = year
        self._geohash = geohash
        self._mean = mean
        self._std = std
        self._count = count
        self._skew = skew
        self._kurtosis = kurtosis

    def get_year(self):
        """Get the year represented by this summary.

        Returns:
            The year of this summary.
        """
        return self._year

    def get_geohash(self):
        """Get the geohash represented by this summary.

        Returns:
            The name of the geohash summarized.
        """
        return self._geohash

    def get_mean(self):
        """Get the mean of observed yields.

        Returns:
            Average yield in this sample.
        """
        return self._mean

    def get_std(self):
        """Get the standard deviation of observed yields.

        Returns:
            Standard deviation of the yield sample.
        """
        return self._std

    def get_count(self):
        """Get the number of observations or data points represented.

        Returns:
            The sample size which may be number of pixels or observations.
        """
        return self._count

    def get_skew(self):
        """Get a measure of skew for this distribution.

        Returns:
            How far off center (0) this distribution was as measured by scipy.stats.skew.
        """
        return self._skew

    def get_kurtosis(self):
        """Get a measure of tail thickness for this distribuiton.

        Returns:
            Tail shape as measured by scipy.stats.kurtosis.
        """
        return self._kurtosis

    def get_key(self):
        """Get a string uniquely representing this geohash / year combination.

        Returns:
            String combining this summary's geohash and year.
        """
        return '%s.%d' % (self.get_geohash(), self.get_year())

    def to_dict(self):
        """Serialize this object to a primitives-only dictionary.

        Returns:
            Serialized version of this record.
        """
        return {
            'year': self.get_year(),
            'geohash': self.get_geohash(),
            'mean': self.get_mean(),
            'std': self.get_std(),
            'count': self.get_count(),
            'skew': self.get_skew(),
            'kurtosis': self.get_kurtosis()
        }

    def combine(self, other):
        """Combine samples between two samples on the same geohash / year combination.

        Args:
            other: The summary with which to combine.

        Returns:
            Summary after combining samples.
        """
        assert self.get_year() == other.get_year()
        assert self.get_geohash() == other.get_geohash()

        self_dist = distribution_struct.Distribution(
            self.get_mean(),
            self.get_std(),
            self.get_count(),
            skew=self.get_skew(),
            kurtosis=self.get_kurtosis()
        )

        other_dist = distribution_struct.Distribution(
            other.get_mean(),
            other.get_std(),
            other.get_count(),
            skew=self.get_skew(),
            kurtosis=self.get_kurtosis()
        )

        new_dist = self_dist.combine(other_dist)

        return GeohashYieldSummary(
            self.get_year(),
            self.get_geohash(),
            new_dist.get_mean(),
            new_dist.get_std(),
            new_dist.get_count(),
            new_dist.get_skew(),
            new_dist.get_kurtosis()
        )


class GeohashYieldSummaryDistDecorator:
    """Summary of distribution of yields found within a geohash for a single year with sample."""

    def __init__(self, inner, sample):
        """Create a new summary.

        Args:
            inner: The GeohashYieldSummary to decorate.
            sample: The sample for the distribution.
        """
        self._inner = inner
        self._sample = sample

    def get_year(self):
        """Get the year represented by this summary.

        Returns:
            The year of this summary.
        """
        return self._inner.get_year()

    def get_geohash(self):
        """Get the geohash represented by this summary.

        Returns:
            The name of the geohash summarized.
        """
        return self._inner.get_geohash()

    def get_mean(self):
        """Get the mean of observed yields.

        Returns:
            Average yield in this sample.
        """
        return self._inner.get_mean()

    def get_std(self):
        """Get the standard deviation of observed yields.

        Returns:
            Standard deviation of the yield sample.
        """
        return self._inner.get_std()

    def get_count(self):
        """Get the number of observations or data points represented.

        Returns:
            The sample size which may be number of pixels or observations.
        """
        return self._inner.get_count()

    def get_skew(self):
        """Get a measure of skew for this distribution.

        Returns:
            How far off center (0) this distribution was as measured by scipy.stats.skew.
        """
        return self._inner.get_skew()

    def get_kurtosis(self):
        """Get a measure of tail thickness for this distribuiton.

        Returns:
            Tail shape as measured by scipy.stats.kurtosis.
        """
        return self._inner.get_kurtosis()

    def get_key(self):
        """Get a string uniquely representing this geohash / year combination.

        Returns:
            String combining this summary's geohash and year.
        """
        return self._inner.get_key()

    def to_dict(self):
        """Serialize this object to a primitives-only dictionary.

        Returns:
            Serialized version of this record.
        """
        base_dict = self._inner.to_dict()
        base_dict['sample'] = ' '.join(map(lambda x: str(x), self._sample))
        return base_dict

    def get_sample_full(self):
        """Get the sample for this distribution.

        Returns:
            Full sample
        """
        return self._sample

    def get_sample(self, count):
        """Sample this distribution.

        Returns:
            Numpy array of random samples.
        """
        return scipy.stats.rvs(
            self._yield_a,
            self._yield_b,
            loc=self._yield_loc,
            scale=self._yield_scale,
            size=count
        )

    def combine(self, other):
        """Combine samples between two samples on the same geohash / year combination.

        Args:
            other: The summary with which to combine.

        Returns:
            Summary after combining samples.
        """
        inner_combined = self._inner.combine(other)

        total_count = self.get_count() + other.get_count()
        percent_self = self.get_count() / total_count
        percent_other = other.get_count() / total_count
        self_sample = self.get_sample(1000 * percent_self)
        other_sample = other.get_sample(1000 * percent_other)
        new_sample = numpy.concatenate([self_sample, other_sample])

        return GeohashYieldSummaryBetaDecorator(
            inner_combined,
            new_sample
        )


class GeohashYieldSummaryBetaDecorator:
    """Summary of distribution of yields found within a geohash for a single year."""

    def __init__(self, inner, yield_a, yield_b, yield_loc, yield_scale):
        """Create a new summary.

        Args:
            inner: The GeohashYieldSummary to decorate.
            yield_a: The a parameter for the beta distribution.
            yield_b: The b parameter for the beta distribution.
            yield_loc: The center or location of the beta distribution.
            yield_scale: The scale of the beta distribution.
        """
        self._inner = inner
        self._yield_a = yield_a
        self._yield_b = yield_b
        self._yield_loc = yield_loc
        self._yield_scale = yield_scale

    def get_year(self):
        """Get the year represented by this summary.

        Returns:
            The year of this summary.
        """
        return self._inner.get_year()

    def get_geohash(self):
        """Get the geohash represented by this summary.

        Returns:
            The name of the geohash summarized.
        """
        return self._inner.get_geohash()

    def get_mean(self):
        """Get the mean of observed yields.

        Returns:
            Average yield in this sample.
        """
        return self._inner.get_mean()

    def get_std(self):
        """Get the standard deviation of observed yields.

        Returns:
            Standard deviation of the yield sample.
        """
        return self._inner.get_std()

    def get_count(self):
        """Get the number of observations or data points represented.

        Returns:
            The sample size which may be number of pixels or observations.
        """
        return self._inner.get_count()

    def get_skew(self):
        """Get a measure of skew for this distribution.

        Returns:
            How far off center (0) this distribution was as measured by scipy.stats.skew.
        """
        return self._inner.get_skew()

    def get_kurtosis(self):
        """Get a measure of tail thickness for this distribuiton.

        Returns:
            Tail shape as measured by scipy.stats.kurtosis.
        """
        return self._inner.get_kurtosis()

    def get_key(self):
        """Get a string uniquely representing this geohash / year combination.

        Returns:
            String combining this summary's geohash and year.
        """
        return self._inner.get_key()

    def to_dict(self):
        """Serialize this object to a primitives-only dictionary.

        Returns:
            Serialized version of this record.
        """
        base_dict = self._inner.to_dict()
        base_dict['a'] = self._yield_a
        base_dict['b'] = self._yield_b
        base_dict['loc'] = self._yield_loc
        base_dict['scale'] = self._yield_scale
        return base_dict

    def get_yield_a(self):
        """Get the beta distribution a parameter.

        Returns:
            Beta distribution a parameter.
        """
        return self._yield_a

    def get_yield_b(self):
        """Get the beta distribution b parameter.

        Returns:
            Beta distribution b parameter.
        """
        return self._yield_b

    def get_yield_loc(self):
        """Get the beta distribution center location.

        Returns:
            Beta distribution center location.
        """
        return self._yield_loc

    def get_yield_scale(self):
        """Get the beta distribution scale.

        Returns:
            Beta distribution scale.
        """
        return self._yield_scale

    def get_sample(self, count):
        """Sample this distribution.

        Returns:
            Numpy array of random samples.
        """
        return scipy.stats.rvs(
            self._yield_a,
            self._yield_b,
            loc=self._yield_loc,
            scale=self._yield_scale,
            size=count
        )

    def combine(self, other):
        """Combine samples between two samples on the same geohash / year combination.

        Args:
            other: The summary with which to combine.

        Returns:
            Summary after combining samples.
        """
        inner_combined = self._inner.combine(other)

        total_count = self.get_count() + other.get_count()
        percent_self = self.get_count() / total_count
        percent_other = other.get_count() / total_count
        self_sample = self.get_sample(1000 * percent_self)
        other_sample = other.get_sample(1000 * percent_other)
        combined = numpy.concatenate([self_sample, other_sample])
        fit_distribution = scipy.stats.beta.fit(combined)

        yield_a = fit_distribution[0]
        yield_b = fit_distribution[1]
        yield_loc = fit_distribution[2]
        yield_scale = fit_distribution[3]

        return GeohashYieldSummaryBetaDecorator(
            inner_combined,
            yield_a,
            yield_b,
            yield_loc,
            yield_scale
        )


class InputTiff:
    """Record of a geotiff (either yield or climate) to be processed."""

    def __init__(self, variable, condition, date):
        """Create a new record for a geotiff containing raw data.

        Args:
            variable: The variable encoded in the geotiff.
            condition: The condition represented by the geotiff which may be historic or future like
                2050_SSP245.
            date: The string date (ISO8601) reprsented by this tiff.
        """
        self._variable = variable
        self._condition = condition
        self._date = date

    def get_variable(self):
        """Get the name of the variable encoded in this geotiff.

        Returns:
            The variable encoded in the geotiff.
        """
        return self._variable

    def get_condition(self):
        """Get the condition in which these data were gathered for which it was generated.

        Returns:
            The condition represented by the geotiff which may be historic or future like
            2050_SSP245.
        """
        return self._condition

    def get_date(self):
        """Get the date from which these data were gathered or for which it was generated.

        Returns:
            The string date (ISO8601) reprsented by this tiff.
        """
        return self._date

    def get_filename(self):
        """Get the name of the geotiff file.

        Returns:
            File name but not full path of the geotiff.
        """
        date_str = self._date.isoformat().replace('-', '.')
        if self._condition == 'observations':
            filename = '%s.%s.tif' % (self._variable, date_str)
            return filename.replace('chirps-v2', 'chirps-v2.0')
        else:
            filename = '%s.%s.%s.tif' % (self._condition, self._variable, date_str)
            return filename.replace('chirps-v2', 'CHIRPS')

    def to_dict(self):
        """Serialize this object as a primitives-only dictionary.

        Returns:
            Serialization as a dict.
        """
        return {
            'variable': self.get_variable(),
            'condition': self.get_condition(),
            'date': self.get_date().isoformat()
        }


def parse_input_tiff(target_dict):
    """Parse a primitives-only dictionary describing an input geotiff.

    Args:
        target_dict: The serialization to parse.

    Returns:
        Parsed record.
    """
    full_datetime = datetime.fromisoformat(target_dict['date'])
    date = datetime.date(full_datetime.year, full_datetime.month, full_datetime.day)
    return InputTiff(
        target_dict['variable'],
        target_dict['condition'],
        date
    )
