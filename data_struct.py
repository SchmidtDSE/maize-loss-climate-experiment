import datetime

import distribution_struct


class GeohashClimateSummary:

    def __init__(self, geohash, year, month, var, condition, mean, std, var_min, var_max, count,
        skew, kurtosis, day=None):
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
        return self._geohash

    def get_year(self):
        return self._year

    def get_month(self):
        return self._month

    def get_var(self):
        return self._var

    def get_condition(self):
        return self._condition

    def get_mean(self):
        return self._mean

    def get_std(self):
        return self._std

    def get_min(self):
        return self._var_min

    def get_max(self):
        return self._var_max

    def get_count(self):
        return self._count

    def get_day(self):
        return self._day

    def get_skew(self):
        return self._skew

    def get_kurtosis(self):
        return self._kurtosis

    def get_without_day(self):
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
        return distribution_struct.Distribution(
            self.get_mean(),
            self.get_std(),
            self.get_count(),
            dist_min=self.get_min(),
            dist_max=self.get_max(),
            skew=self.get_skew(),
            kurtosis=self.get_kurtosis()
        )

    def combine(self, other):
        self_dist = self.get_distribution()
        other_dist = other.get_distribution()
        new_dist = self_dist.combine(other_dist)

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

    def __init__(self, year, geohash, mean, std, count, skew, kurtosis):
        self._year = year
        self._geohash = geohash
        self._mean = mean
        self._std = std
        self._count = count
        self._skew = skew
        self._kurtosis = kurtosis

    def get_year(self):
        return self._year

    def get_geohash(self):
        return self._geohash

    def get_key(self):
        return '%s.%d' % (self.get_geohash(), self.get_year())

    def get_mean(self):
        return self._mean

    def get_std(self):
        return self._std

    def get_count(self):
        return self._count

    def get_skew(self):
        return self._skew

    def get_kurtosis(self):
        return self._kurtosis

    def to_dict(self):
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


def parse_geohash_yield_summary(target_dict):
    return GeohashYieldSummary(
        int(target_dict['year']),
        target_dict['geohash'],
        target_dict['mean'],
        target_dict['std'],
        target_dict['count'],
        target_dict['skew'],
        target_dict['kurtosis']
    )


class InputTiff:

    def __init__(self, variable, condition, date):
        self._variable = variable
        self._condition = condition
        self._date = date

    def get_variable(self):
        return self._variable

    def get_condition(self):
        return self._condition

    def get_date(self):
        return self._date

    def get_filename(self):
        date_str = self._date.isoformat().replace('-', '.')
        if self._condition == 'observations':
            filename = '%s.%s.tif' % (self._variable, date_str)
            return filename.replace('chirps-v2', 'chirps-v2.0')
        else:
            filename = '%s.%s.%s.tif' % (self._condition, self._variable, date_str)
            return filename.replace('chirps-v2', 'CHIRPS')

    def to_dict(self):
        return {
            'variable': self.get_variable(),
            'condition': self.get_condition(),
            'date': self.get_date().isoformat()
        }


def parse_input_tiff(target_dict):
    full_datetime = datetime.fromisoformat(target_dict['date'])
    date = datetime.date(full_datetime.year, full_datetime.month, full_datetime.day)
    return InputTiff(
        target_dict['variable'],
        target_dict['condition'],
        date
    )
