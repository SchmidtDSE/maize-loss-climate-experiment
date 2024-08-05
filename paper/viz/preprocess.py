import toolz.itertoolz

import const


class InterpretedRecord:

    def get_count(self):
        raise NotImplementedError('Use implementor.')

    def get_category(self):
        raise NotImplementedError('Use implementor.')


class Precent:

    def __init__(self, name, percent):
        self._name = name
        self._percent = percent

    def get_name(self):
        return self._name

    def get_percent(self):
        return self._percent


def make_percents(interpreted_records):
    total_count = sum(map(lambda x: x.get_count(), interpreted_records))
    new_records = map(
        lambda x: {'category': x.get_category(), 'count': x.get_count()},
        interpreted_records
    )
    new_records_reduced = toolz.itertoolz.reduceby(
        lambda x: x['category'],
        lambda a, b: {
            'category': a['category'],
            'count': a['count'] + b['count']
        },
        new_records
    )
    reduced_flat = new_records_reduced.values()
    percents = map(
        lambda x: Precent(x['category'], x['count'] / total_count),
        reduced_flat
    )
    return sorted(percents, key=lambda x: x.get_percent(), reverse=True)


class ScatterPoint(InterpretedRecord):

    def __init__(self, geohash, x_value, y_value, count, category, latitude, longitude):
        self._geohash = geohash
        self._x_value = x_value
        self._y_value = y_value
        self._count = count
        self._category = category
        self._latitude = latitude
        self._longitude = longitude

    def get_geohash(self):
        return self._geohash

    def get_x_value(self):
        return self._x_value

    def get_y_value(self):
        return self._y_value

    def get_count(self):
        return self._count

    def get_category(self):
        return self._category

    def get_latitude(self):
        return self._latitude

    def get_longitude(self):
        return self._longitude

    def get_is_valid(self):
        required_values = [
            self._geohash,
            self._x_value,
            self._y_value,
            self._count,
            self._category,
            self._latitude,
            self._longitude
        ]
        nones = filter(lambda x: x is None, required_values)
        num_nones = sum(map(lambda x: 1, nones))
        return num_nones == 0


def get_scenario_year(record):
    scenario = record.get_scenario()
    if scenario.startswith('2030'):
        return 2030
    elif scenario.startswith('2050'):
        return 2050
    else:
        return 'unknown'


def make_scatter_values(records, climate_deltas, configuration):
    scenario = configuration.get_scenario()
    target_year = int(scenario[:4])
    target_loss = '25% loss' if configuration.get_loss() == '75% cov' else '15% loss'
    scenario_records = filter(lambda x: scenario.startswith(str(get_scenario_year(x))), records)
    loss_records = filter(lambda x: x.get_loss() == target_loss, scenario_records)

    if configuration.get_risk_range() == 'Sample 1 Year':
        year_records = filter(lambda x: x.get_year() in [2030, 2050], loss_records)
        count_multiplier = 1  # single year summarized
    else:
        year_records_nested = toolz.itertoolz.reduceby(
            lambda x: x.get_key(),
            lambda a, b: a.combine(b),
            loss_records
        )
        year_records = year_records_nested.values()
        count_multiplier = 5  # 5 years summarized

    year_records = list(year_records)
    count = len(year_records) * count_multiplier

    p_threshold_naive = {
        'p <  0.05': 0.05,
        'p <  0.10': 0.1
    }[configuration.get_threshold()]

    if configuration.get_adjustment() == 'Bonferroni':
        p_threshold = p_threshold_naive / count
    else:
        p_threshold = p_threshold_naive

    month_num = const.MONTH_NUMS[configuration.get_month()]

    def get_climate_delta(record):
        geohash = record.get_geohash()
        return climate_deltas.get(geohash, target_year, month_num)

    get_var_x = {
        'no var': lambda x: None,
        'chirps': lambda x: get_climate_delta(x).get_chirps(),
        'rhn': lambda x: get_climate_delta(x).get_rhn(),
        'rhx': lambda x: get_climate_delta(x).get_rhx(),
        'svp': lambda x: get_climate_delta(x).get_svp(),
        'tmax': lambda x: get_climate_delta(x).get_tmax(),
        'tmin': lambda x: get_climate_delta(x).get_tmin(),
        'vpd': lambda x: get_climate_delta(x).get_vpd(),
        'wbgtmax': lambda x: get_climate_delta(x).get_wbgtmax()
    }[configuration.get_var()]

    def make_point_yield(record):
        prior_mean = record.get_yield_comparison().get_prior().get_mean()
        predicted_mean = record.get_yield_comparison().get_predicted().get_mean()
        percent_change = predicted_mean - prior_mean
        p_value = record.get_yield_comparison().get_p_value()

        if p_value > p_threshold:
            category = 'no significant change'
        elif prior_mean > predicted_mean:
            category = 'lower than counterfactual'
        else:
            category = 'higher than counterfactual'

        var_x = get_var_x(record)
        effective_x = prior_mean if var_x is None else var_x
        effective_y = predicted_mean if var_x is None else percent_change

        return ScatterPoint(
            record.get_geohash(),
            effective_x,
            effective_y,
            record.get_num(),
            category,
            record.get_latitude(),
            record.get_longitude()
        )

    def make_point_risk(record):
        risk_change = record.get_predicted_risk().get_risk_change()
        risk_p = record.get_predicted_risk().get_p_value()

        after_yield = record.get_yield_comparison().get_predicted().get_mean()
        before_yield = record.get_yield_comparison().get_prior().get_mean()
        yield_change = after_yield - before_yield

        if yield_change > 0:
            var_str = 'yield above counterfactual'
        else:
            var_str = 'yield below counterfactual'

        if risk_p > p_threshold:
            category = 'no significant change'
        elif risk_change > 0:
            category = 'higher risk, ' + var_str
        else:
            category = 'lower risk, ' + var_str

        var_x = get_var_x(record)
        effective_x = yield_change if var_x is None else var_x

        return ScatterPoint(
            record.get_geohash(),
            effective_x,
            risk_change * 100,
            record.get_num(),
            category,
            record.get_latitude(),
            record.get_longitude()
        )

    def make_point_adapt(record):
        predicted_change = record.get_predicted_risk().get_risk_change()
        predicted_p = record.get_predicted_risk().get_p_value()

        adapted_change = record.get_adapted_risk().get_risk_change()
        adapted_p = record.get_adapted_risk().get_p_value()

        if predicted_p > p_threshold:
            category = 'no significant change'
        elif predicted_change > 0:
            if adapted_p < p_threshold and adapted_change < predicted_change:
                category = 'higher risk, can adapt'
            else:
                category = 'higher risk, cant adapt'
        else:
            if adapted_p < p_threshold and adapted_change < predicted_change:
                category = 'lower risk, can adapt'
            else:
                category = 'lower risk, cant adapt'

        var_x = get_var_x(record)
        effective_x = adapted_change * 100 if var_x is None else var_x
        net_change = predicted_change + adapted_change
        effective_y = predicted_change * 100 if var_x is None else net_change * 100

        return ScatterPoint(
            record.get_geohash(),
            effective_x,
            effective_y,
            record.get_num(),
            category,
            record.get_latitude(),
            record.get_longitude()
        )

    if configuration.get_metric() == 'yield':
        make_point = make_point_yield
    elif configuration.get_metric() == 'risk':
        make_point = make_point_risk
    else:
        make_point = make_point_adapt

    points = map(make_point, year_records)
    points_valid = filter(lambda x: x.get_is_valid(), points)

    if configuration.get_sig_filter() == 'significant only':
        points_filtered = filter(
            lambda x: x.get_category() != 'no significant change',
            points_valid
        )
    else:
        points_filtered = points_valid

    return list(points_filtered)
