import math


class PlacedRecord:

    def __init__(self, x, y, record):
        self._x = x
        self._y = y
        self._record = record

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def get_record(self):
        return self._record

    def in_range(self, x_center, y_center, radius_allowed):
        radius = math.sqrt((self._x - x_center)**2 + (self._y - y_center)**2)
        return radius < radius_allowed


class RiskComparison:

    def __init__(self, control_risk, exprimental_risk, p_value, count):
        self._control_risk = control_risk
        self._experimental_risk = exprimental_risk
        self._p_value = p_value
        self._count = count

    def get_control_risk(self):
        return self._control_risk

    def get_experimental_risk(self):
        return self._experimental_risk

    def get_risk_change(self):
        return self._experimental_risk - self._control_risk

    def get_p_value(self):
        return self._p_value

    def get_count(self):
        return self._count

    def combine(self, other):
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

    def __init__(self, mean, std, count):
        self._mean = mean
        self._std = std
        self._count = count

    def get_mean(self):
        return self._mean

    def get_std(self):
        return self._std

    def get_std_percent(self):
        return self._std / self._mean

    def get_count(self):
        return self._count

    def combine(self, other):
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

    def __init__(self, prior, predicted, p_value):
        self._prior = prior
        self._predicted = predicted
        self._p_value = p_value

    def get_prior(self):
        return self._prior

    def get_predicted(self):
        return self._predicted

    def get_p_value(self):
        return self._p_value

    def combine(self, other):
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

    def __init__(self, geohash, year, scenario, num, predicted_risk, adapted_risk,
        yield_comparison, latitude, longitude, loss):
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
        return self._geohash

    def get_year(self):
        return self._year

    def get_scenario(self):
        return self._scenario

    def get_num(self):
        return self._num

    def get_predicted_risk(self):
        return self._predicted_risk

    def get_adapted_risk(self):
        return self._adapted_risk

    def get_yield_comparison(self):
        return self._yield_comparison

    def get_loss(self):
        return self._loss

    def get_key(self):
        pieces = [
            self.get_geohash(),
            self.get_scenario(),
            self.get_loss()
        ]
        pieces_str = map(lambda x: str(x), pieces)
        return '\t'.join(pieces_str)

    def get_latitude(self):
        return self._latitude

    def get_longitude(self):
        return self._longitude

    def combine(self, other):
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
    counterfactual_risk_rate = float(raw_record['counterfactualRisk'])
    predicted_risk_rate = float(raw_record['predictedRisk'])
    adapted_risk_rate = predicted_risk_rate  # float(raw_record['adaptedRisk'])
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
        float(raw_record['p']),  # float(raw_record['pAdaptedRisk']),
        num
    )

    original_yield = YieldDistribution(
        float(raw_record['counterfactualMean']),
        0,  # float(raw_record['counterfactualStd']),
        num
    )

    predicted_yield = YieldDistribution(
        float(raw_record['predictedMean']),
        0,  # float(raw_record['predictedYieldStd']),
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

    def __init__(self, geohash, year, month, rhn, rhx, tmax, tmin, chirps, svp, vpd, wbgtmax):
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
        return self._year

    def get_geohash(self):
        return self._geohash

    def get_month(self):
        return self._month

    def get_rhn(self):
        return self._rhn

    def get_rhx(self):
        return self._rhx

    def get_tmax(self):
        return self._tmax

    def get_tmin(self):
        return self._tmin

    def get_chirps(self):
        return self._chirps

    def get_svp(self):
        return self._svp

    def get_vpd(self):
        return self._vpd

    def get_wbgtmax(self):
        return self._wbgtmax

    def get_key(self):
        return get_climate_delta_key(self._geohash, self._year, self._month)


def try_get_float(target):
    if target.strip() == '':
        return None
    else:
        return float(target)


def parse_climate_delta(raw_record):
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
    pieces = [geohash, year, month]
    pieces_str = map(lambda x: str(x), pieces)
    return '\t'.join(pieces_str)


class ClimateDeltas:

    def __init__(self, inner_deltas):
        self._indexed = dict(map(lambda x: (x.get_key(), x), inner_deltas))

    def get(self, geohash, year, month):
        key = get_climate_delta_key(geohash, year, month)
        return self._indexed[key]
