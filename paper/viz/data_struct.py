"""Data structures for more complex types.

License:
    BSD
"""

import math


class PlacedRecord:
    """Record with an x, y coordinate associated."""

    def __init__(self, x, y, record):
        """Create a new record with a x, y coordinate location.

        Args:
            x: The horizontal coordinate of the record.
            y: The vertical coordinate of the record.
            record: The record to which the coordinates are associated.
        """
        self._x = x
        self._y = y
        self._record = record

    def get_x(self):
        """Get the horizontal coordinate of this record.

        Returns:
            The x coordinate in pixels.
        """
        return self._x

    def get_y(self):
        """Get the vertical coordinate of this record.

        Returns:
            The y coordinate in pixels.
        """
        return self._y

    def get_record(self):
        """Get the record which has been placed.

        Returns:
            The record with which the coordinates are associated.
        """
        return self._record

    def in_range(self, x_center, y_center, radius_allowed):
        """Determine if this record is within distance to a reference point.

        Args:
            x_center: The horizontal coordinate of the reference point.
            y_center: The vertical coordinate of the reference point.
            radius_allowed: The max distance in pixels from the reference point.

        Returns:
            True if this record is within the radius_allows from the reference point and false
            otherwise.
        """
        radius = math.sqrt((self._x - x_center)**2 + (self._y - y_center)**2)
        return radius < radius_allowed


class RiskComparison:
    """Record in which risks are compared between a control and experimental group."""

    def __init__(self, control_risk, exprimental_risk, p_value, count):
        """Make a new risk comparison record.

        Args:
            control_risk: The claims rate for the control group.
            experimental_risk: The claims rate for the expeirmental group.
            p_value: The p value associated with the difference between the groups.
            count: The sample size associated with this comparison.
        """
        self._control_risk = control_risk
        self._experimental_risk = exprimental_risk
        self._p_value = p_value
        self._count = count

    def get_control_risk(self):
        """Get the loss risk associated with the control group.

        Returns:
            The claims rate for the control group.
        """
        return self._control_risk

    def get_experimental_risk(self):
        """Get the loss risk associated with the experimental group.

        Returns:
            The claims rate for the expeirmental group.
        """
        return self._experimental_risk

    def get_risk_change(self):
        """Get the difference between experimental risk and control risk.

        Returns:
            The difference from the control loss risk to the experimental loss risk.
        """
        return self._experimental_risk - self._control_risk

    def get_p_value(self):
        """Get the p value associated with the statistical test comparing these results.

        Returns:
            The p value associated with the difference between the groups.
        """
        return self._p_value

    def get_count(self):
        """Get the sample size used in this comparison's statistical test.

        Returns:
            The sample size associated with this comparison.
        """
        return self._count

    def combine(self, other):
        """Combine two samples together (like geographic aggregation).

        Combine two samples together (like geographic aggregation) in which samples are pooled and
        the minimum of p values is used (neighborhoods with any year or components with significant
        changes).

        Args:
            other: The other result with which to pool.

        Returns:
            Comparison after pooling.
        """
        self_count = self.get_count()
        other_count = other.get_count()

        if self_count == 0:
            return other
        elif other_count == 0:
            return self

        # We are highlighting neighborhoods with any year with significant changes
        new_p = min([self.get_p_value(), other.get_p_value()])

        def combine_probs(a, a_count, b, b_count):
            return (a * a_count + b * b_count) / (a_count + b_count)

        self_control_risk = self.get_control_risk()
        other_control_risk = other.get_control_risk()
        new_control_risk = combine_probs(
            self_control_risk,
            self_count,
            other_control_risk,
            other_count
        )

        self_experimental_risk = self.get_experimental_risk()
        other_experimental_risk = other.get_experimental_risk()
        new_experimental_risk = combine_probs(
            self_experimental_risk,
            self_count,
            other_experimental_risk,
            other_count
        )

        new_count = self_count + other_count

        return RiskComparison(
            new_control_risk,
            new_experimental_risk,
            new_p,
            new_count
        )


class YieldDistribution:
    """Record of a distribution of yields."""

    def __init__(self, mean, std, count):
        """Create a new yield distribution record.

        Args:
            mean: The average of yields.
            std: The standard deivation of yields.
            count: The sample size represented by this distribution.
        """
        self._mean = mean
        self._std = std
        self._count = count

    def get_mean(self):
        """Get the mean of this yield distribution.

        Returns:
            The average of yields.
        """
        return self._mean

    def get_std(self):
        """Get the standard deviation of this yield distribution.

        Returns:
            The standard deivation of yields.
        """
        return self._std

    def get_std_percent(self):
        """Get the ratio of standard deviation to mean.

        Returns:
            Result of dividing standard deviation by mean.
        """
        return self._std / self._mean

    def get_count(self):
        """Get the sample size associated with this distribution.

        Returns:
            The sample size represented by this distribution.
        """
        return self._count

    def combine(self, other):
        """Pool the samples of this distribution with another distribution.

        Args:
            other: The other distribution with which to pool.

        Returns:
            New distribution after pooling samples.
        """
        self_count = self.get_count()
        other_count = other.get_count()

        if self_count == 0:
            return other
        elif other_count == 0:
            return self

        self_weighted_mean = self.get_mean() * self_count
        other_weighted_mean = other.get_mean() * other_count
        new_mean = (self_weighted_mean + other_weighted_mean) / (self_count + other_count)

        self_var_piece = (self_count - 1) * self.get_std()**2
        other_var_piece = (other_count - 1) * other.get_std()**2
        pooled_count = self_count + other_count - 2

        if pooled_count == 0:
            new_std = (self.get_std() + other.get_std()) / 2
        else:
            new_std = math.sqrt((self_var_piece + other_var_piece) / pooled_count)

        new_count = self_count + other_count
        return YieldDistribution(new_mean, new_std, new_count)


class YieldComparison:
    """Record comparing either an aggregate or average yield before and after a change or delay."""

    def __init__(self, prior, predicted, p_value):
        """Create a new comparison record.

        Args:
            prior: The preivous yield level.
            predicted: The yield level after the change or delay.
            p_value: The p value associated with the statistical test comparing before and after.
        """
        self._prior = prior
        self._predicted = predicted
        self._p_value = p_value

    def get_prior(self):
        """Get the yield prior to the change or delay.

        Returns:
            The preivous yield level.
        """
        return self._prior

    def get_predicted(self):
        """Get the yield level after the change or delay.

        Returns:
            The yield level after the change or delay.
        """
        return self._predicted

    def get_p_value(self):
        """Get the p value of the statistical test evalating the change or delay.

        Returns:
            The p value associated with the statistical test comparing before and after.
        """
        return self._p_value

    def combine(self, other):
        """Pool changes.

        Pool changes such that the minimum of the two p values is used to represent any component
        with a statistical change.

        Returns:
            Description of the change after combining samples.
        """
        self_count = self.get_predicted().get_count()
        other_count = other.get_predicted().get_count()

        if self_count == 0:
            return other
        elif other_count == 0:
            return self

        new_p = min([self.get_p_value(), other.get_p_value()])
        return YieldComparison(
            self.get_prior().combine(other.get_prior()),
            self.get_predicted().combine(other.get_predicted()),
            new_p
        )


class Record:
    """Record of an individual neighborhood / geohash."""

    def __init__(self, geohash, year, scenario, num, predicted_risk, adapted_risk,
        yield_comparison, latitude, longitude, loss):
        """Create a new record.

        Args:
            geohash: The geohash name (string) of the area represented.
            year: The year for which data are reported.
            scenario: The scenario from which the data are provided or historical.
            num: The count of observations within this neighborhood.
            predicted_risk: The loss probability expected.
            adapted_risk: The loss probability anticipated with adaptation steps taken.
            yield_comparison: Record describing how the yields differ between the historic values
                and the expected values.
            latitude: The latitude of the center of this neighborhood in degrees.
            longitude: The longitude of the center of this neighborhood in degrees.
            loss: The loss level / loss threshold evaluated.
        """
        self._geohash = geohash
        self._year = year
        self._scenario = scenario
        self._num = num
        self._predicted_risk = predicted_risk
        self._adapted_risk = adapted_risk
        self._yield_comparison = yield_comparison
        self._loss = loss
        self._latitude = latitude
        self._longitude = longitude

    def get_geohash(self):
        """Get the geohash represented by this record.

        Returns:
            The geohash name (string) of the area represented.
        """
        return self._geohash

    def get_year(self):
        """Get the year associated with this observed or modeled data.

        Returns:
            The year for which data are reported.
        """
        return self._year

    def get_scenario(self):
        """Get the scenario name (string) in which these data are found.

        Returns:
            The scenario from which the data are provided or historical.
        """
        return self._scenario

    def get_num(self):
        """Get the sample size associated with this neighborhood.

        Returns:
            The count of observations within this neighborhood.
        """
        return self._num

    def get_predicted_risk(self):
        """Get the exepected risk level as a float.

        Returns:
            The loss probability expected (0 - 1).
        """
        return self._predicted_risk

    def get_adapted_risk(self):
        """Get the risk anticiapted with adaptation as a float.

        Returns:
            The loss probability anticipated with adapation (0 - 1).
        """
        return self._adapted_risk

    def get_yield_comparison(self):
        """Get a record describing how yields are expected to change (not in adapated case).

        Returns:
            Record describing how the yields differ between the historic values and the expected
            values.
        """
        return self._yield_comparison

    def get_loss(self):
        """Get the loss threshold evaluated in this neighborhood.

        Returns:
            The loss level / loss threshold evaluated (0 - 1).
        """
        return self._loss

    def get_key(self):
        """Get a key uniquely describing this geohash within simulation.

        Returns:
            Key combining geohash name, scenario, and loss level evaluated.
        """
        pieces = [
            self.get_geohash(),
            self.get_scenario(),
            self.get_loss()
        ]
        pieces_str = map(lambda x: str(x), pieces)
        return '\t'.join(pieces_str)

    def get_latitude(self):
        """Get the latitude associated with this record.

        Returns:
            The latitude of the center of this neighborhood in degrees.
        """
        return self._latitude

    def get_longitude(self):
        """Get the longitude associated with this record.

        Returns:
            The longitude of the center of this neighborhood in degrees.
        """
        return self._longitude

    def combine(self, other):
        """Pool samples from different records, potentially across years.

        Args:
            other: The record with which to pool.

        Returns:
            New record representing the pooling of this and other.
        """
        assert self.get_key() == other.get_key()
        return Record(
            self.get_geohash(),
            (self.get_year() + other.get_year()) / 2,
            self.get_scenario(),
            self.get_num() + other.get_num(),
            self.get_predicted_risk().combine(other.get_predicted_risk()),
            self.get_adapted_risk().combine(other.get_adapted_risk()),
            self.get_yield_comparison().combine(other.get_yield_comparison()),
            self.get_latitude(),
            self.get_longitude(),
            self.get_loss()
        )


def parse_record(raw_record):
    """Parse a Record object from a raw primitives-only dictionary.

    Args:
        raw_record: The raw record to parse.

    Returns:
        Record after parsing with expected types.
    """
    def try_float(target):
        if target == '':
            return None
        else:
            return float(target)

    counterfactual_risk_rate = try_float(raw_record['counterfactualRisk'])
    predicted_risk_rate = try_float(raw_record['predictedRisk'])
    adapted_risk_rate = predicted_risk_rate
    num = round(float(raw_record['num']))

    predicted_risk = RiskComparison(
        counterfactual_risk_rate,
        predicted_risk_rate,
        float(raw_record['p']),
        num
    )

    adapted_risk = RiskComparison(
        predicted_risk_rate,
        adapted_risk_rate,
        float(raw_record['p']),
        num
    )

    original_yield = YieldDistribution(
        float(raw_record['counterfactualMean']),
        0,
        num
    )

    predicted_yield = YieldDistribution(
        float(raw_record['predictedMean']),
        0,
        num
    )

    yield_comparison = YieldComparison(
        original_yield,
        predicted_yield,
        float(raw_record['p'])
    )

    geohash = str(raw_record['geohash'])
    latitude = float(raw_record['lat'])
    longitude = float(raw_record['lng'])
    year = int(raw_record['year'])
    condition = str(raw_record['condition'])
    loss = '%d%% loss' % round(float(raw_record['lossThreshold']) * 100)

    return Record(
        geohash,
        year,
        condition,
        num,
        predicted_risk,
        adapted_risk,
        yield_comparison,
        latitude,
        longitude,
        loss
    )


class ClimateDelta:
    """Record describing how climate changed in an area."""

    def __init__(self, geohash, year, month, rhn, rhx, tmax, tmin, chirps, svp, vpd, wbgtmax):
        """Create a new record of how growing conditions changed in an area.

        Args:
            geohash: The geohash for which the change is provided.
            year: The year in which the change is expected or was observed.
            month: The month in which the change is expected or was observed.
            rhn: Change in relative humnidity (n) measured in z (num std).
            rhx: Change in relative humidity (x) measured in z (num std).
            tmax: Change in max temperature measured in z (num std).
            tmin: Change in min temperature measured in z (num std).
            chirps: Change in preciptation measured in z (num std).
            svp: Change in saturation vapor pressure in z (num std).
            vpd: Change in vapor pressure deficit in z (num std).
            wbgtmax: Change in wet bulb temperature (max) in z (num std).
        """
        self._year = year
        self._geohash = geohash
        self._month = month
        self._rhn = rhn
        self._rhx = rhx
        self._tmax = tmax
        self._tmin = tmin
        self._chirps = chirps
        self._svp = svp
        self._vpd = vpd
        self._wbgtmax = wbgtmax

    def get_year(self):
        """Get the year in which this change is expected.

        Returns:
            The year in which the change is expected or was observed.
        """
        return self._year

    def get_geohash(self):
        """Get the geohash in which these changes are expected.

        Returns:
            The geohash for which the change is provided.
        """
        return self._geohash

    def get_month(self):
        """Get the string month (like jan) in which these changes are expected.

        Returns:
            The month in which the change is expected or was observed.
        """
        return self._month

    def get_rhn(self):
        """Get anticipated change in daily relative humidity (n).

        Returns:
            Change in relative humnidity (n) measured in z (num std).
        """
        return self._rhn

    def get_rhx(self):
        """Get anticipated change in daily relative humidity (x).

        Returns:
            Change in relative humidity (x) measured in z (num std).
        """
        return self._rhx

    def get_tmax(self):
        """Get the anticipated change in maximum daily temperature.

        Returns:
            Change in max temperature measured in z (num std).
        """
        return self._tmax

    def get_tmin(self):
        """Get the anticipated change in minimum daily temperature.

        Returns:
            Change in min temperature measured in z (num std).
        """
        return self._tmin

    def get_chirps(self):
        """Get the anticipated change in daily precipitation.

        Returns:
            Change in preciptation measured in z (num std).
        """
        return self._chirps

    def get_svp(self):
        """Get the anticipated change in daily SVP.

        Returns:
            Change in saturation vapor pressure in z (num std).
        """
        return self._svp

    def get_vpd(self):
        """Get the anticipated change in daily VPD.

        Returns:
            Change in vapor pressure deficit in z (num std).
        """
        return self._vpd

    def get_wbgtmax(self):
        """Get the anticipated change in daily wet bulb temperature max.

        Returns:
            Change in wet bulb temperature (max) in z (num std).
        """
        return self._wbgtmax

    def get_key(self):
        """Get a key uniquely representing this geohash in time.

        Returns:
            Key uniquely identifying this geohash, year, month combination.
        """
        return get_climate_delta_key(self._geohash, self._year, self._month)


def try_get_float(target):
    """Try parsing a string as a float or return None if empty string.

    Try parsing a string as a float or return None if empty string, throwing an error if parsing
    fails.

    Args:
        target: The value to parse.

    Raises:
        ValueError: Raised if the non-empty string cannot be parsed.

    Returns:
        None if empty string, otherwise stringp parsed as float.
    """
    if target.strip() == '':
        return None
    else:
        return float(target)


def parse_climate_delta(raw_record):
    """Parse a climate delta record from a primitives only record.

    Args:
        raw_record: The record (primitives-only dictionary) to parse.

    Returns:
        ClimateDelta after parsing.
    """
    return ClimateDelta(
        str(raw_record['geohash']),
        int(raw_record['year']),
        int(raw_record['month']),
        try_get_float(raw_record['rhnMeanChange']),
        try_get_float(raw_record['rhxMeanChange']),
        try_get_float(raw_record['tmaxMeanChange']),
        try_get_float(raw_record['tminMeanChange']),
        try_get_float(raw_record['chirpsMeanChange']),
        try_get_float(raw_record['svpMeanChange']),
        try_get_float(raw_record['vpdMeanChange']),
        try_get_float(raw_record['wbgtmaxMeanChange'])
    )


def get_climate_delta_key(geohash, year, month):
    """Get key uniquely identifying a geohash in time.

    Returns:
        Key uniquely identifying this geohash, year, month combination.
    """
    pieces = [geohash, year, month]
    pieces_str = map(lambda x: str(x), pieces)
    return '\t'.join(pieces_str)


class ClimateDeltas:
    """Collection of records describing changes to growing conditions."""

    def __init__(self, inner_deltas):
        """Create a new collection of climate deltas."""
        self._indexed = dict(map(lambda x: (x.get_key(), x), inner_deltas))

    def get(self, geohash, year, month):
        """Get a climate delta.

        Args:
            geohash: The area to look up given its geohash name.
            year: The integer year in which to lookup that geohash.
            month: The integer month in which to lookup thta geohash.

        Raises:
            IndexError: Raised if results matching the request cannot be found.

        Returns:
            ClimateDelta matching the query.
        """
        key = get_climate_delta_key(geohash, year, month)
        return self._indexed[key]
