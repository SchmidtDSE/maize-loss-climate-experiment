"""Structures for preprocessing in the neighborhood-level geohash visualization.

License:
    BSD
"""

import toolz.itertoolz

import const


class InterpretedRecord:
    """Interface for a count or group size to be visualized."""

    def get_count(self):
        """Get the size of the group.
        
        Returns:
            Get the number of instances in this group or its size.
        """
        raise NotImplementedError('Use implementor.')

    def get_category(self):
        """Get the name of the category for which a size is reported.
        
        Returns:
            String human readable category name.
        """
        raise NotImplementedError('Use implementor.')


class Precent:
    """Record of a named percentage."""

    def __init__(self, name, percent):
        """Create a new Percent record.
        
        Args:
            name: The name of the group whose size is to be reported as a percentage.
            percent: The size of the group as a percentage (0 - 1).
        """
        self._name = name
        self._percent = percent

    def get_name(self):
        """Get the name of the group.
        
        Returns:
            The name of the group whose size is reported as a percentage.
        """
        return self._name

    def get_percent(self):
        """Get the size of this group.
        
        Returns:
            The size of the group as a percentage (0 - 1).
        """
        return self._percent


def make_percents(interpreted_records):
    """Convert a group of interpreted records into percentages.
    
    Args:
        interpreted_records: Collection of InterpretedRecord to convert to percentages.
    
    Returns:
        Collection of Precent.
    """
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
    """Form of a Record with information needed to be displayed in a scatterplot or map."""

    def __init__(self, geohash, x_value, y_value, count, category, latitude, longitude):
        """Create a new scatterplot point record.
        
        Args:
            geohash: The name of the geohash represented by a point in a scatterplot or map.
            x_value: The metric to be displayed on the horizontal axis / dictating horizontal
                position.
            y_value: The metric to be displayed on the vertical axis / dictating vertical position.
            count: The size of the group for the scatterplot point.
            category: A description of the subpopulation like 'higher than counterfactual'
                that this point is part of.
            latitude: The geospatial latitude of the center point of this neighborhood.
            longitude: The geospatial longitude of the center point of this neighborhood.
        """
        self._geohash = geohash
        self._x_value = x_value
        self._y_value = y_value
        self._count = count
        self._category = category
        self._latitude = latitude
        self._longitude = longitude

    def get_geohash(self):
        """Get the geohash represented by this point.
        
        Returns:
            The name of the geohash represented by a point in a scatterplot or map.
        """
        return self._geohash

    def get_x_value(self):
        """Get the metric dictating the horizontal position of this point.
        
        Returns:
            The metric to be displayed on the horizontal axis / dictating horizontal position.
        """
        return self._x_value

    def get_y_value(self):
        """Get the metric dictating the vertical position of this point.
        
        Returns:
            The metric to be displayed on the vertical axis / dictating vertical position.
        """
        return self._y_value

    def get_count(self):
        """Get the size of this point.
        
        Returns:
            The size of the group for the scatterplot point either as a count or proportional
            sample weight as float.
        """
        return self._count

    def get_category(self):
        """Get the name of the category to which this point belongs.
        
        Returns:
            A description of the subpopulation like 'higher than counterfactual' that this point is
            part of.
        """
        return self._category

    def get_latitude(self):
        """Get the center latitude of this group.
        
        Returns:
            The geospatial latitude of the center point of this neighborhood.
        """
        return self._latitude

    def get_longitude(self):
        """Get the center longitude of this group.
        
        Returns:
            The geospatial longitude of the center point of this neighborhood.
        """
        return self._longitude

    def get_is_valid(self):
        """Determine if this record is missing any values.
        
        Returns:
            True if no values are missing and false otherwise.
        """
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
    """Get the year series for which results are provided.
    
    Returns:
        Integer year describing the timeseries or 'unknown' if it could not be determined.
    """
    scenario = record.get_scenario()
    if scenario.startswith('2030'):
        return 2030
    elif scenario.startswith('2050'):
        return 2050
    else:
        return 'unknown'


def make_scatter_values(records, climate_deltas, configuration):
    """Convert Records to ScatterPoints.
    
    Args:
        records: Collection of Records to convert to ScatterPoints.
        climate_deltas: Information about how climate is expected to change.
        configuration: The Configuration in which the ScatterPoints will operate.
    
    Returns:
        List of ScatterPoints.
    """
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
