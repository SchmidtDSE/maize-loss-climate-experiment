"""Widgets to configure / manage the geohash-level results visualization.

License:
    BSD
"""
import buttons
import const


class Configuration:
    """Data structure representing the geohash-level results visualization configuration state."""

    def __init__(self, scenario, risk_range, metric, visualization, threshold,
                 adjustment, sig_filter, var, month, loss):
        """Create a new record of configuration.

        Args:
            scenario: The scenario selected like 2050 series.
            risk_range: Indication of if using single year sampling or not (Avg All Years or Sample
                1 Year).
            metric: The metric to display (yield or risk).
            visualization: The visualization type to display (scatter or map).
            threshold: Threshold for significance like p <  0.05.
            adjustment: Indication of if "Bonferroni" or "no correction" should be applied in
                determining significance.
            sig_filter: Indication of if only significant results (significant only) or all results
                (all) should be displayed.
            var: The variable to use to contextualize the metric like chirps. May also be 'no var'
                in which case no dimension contextualizes the metric.
            month: The string name of the month (like jan) from which to display contextual
                dimensional data. Ignored if no context var selected.
            loss: The loss level epxressed as a level of coverage (75% cov, 85% cov).
        """
        self._scenario = scenario
        self._risk_range = risk_range
        self._metric = metric
        self._threshold = threshold
        self._adjustment = adjustment
        self._sig_filter = sig_filter
        self._var = var
        self._month = month
        self._loss = loss

        self._set_viz(visualization)

    def get_scenario(self):
        """Get the scenario selected by the user.

        Returns:
            The scenario selected like 2050 series.
        """
        return self._scenario

    def get_with_scenario(self, new_val):
        """Make a copy of this configuration with a new scenario selection.

        Args:
            new_val: The new scenario.

        Returns:
            A copy of this configuration but with the new selected scenario.
        """
        return Configuration(new_val, self._risk_range, self._metric,
                             self._visualization, self._threshold,
                             self._adjustment, self._sig_filter, self._var,
                             self._month, self._loss)

    def get_risk_range(self):
        """Get the year range which should be used in determining risk.

        Returns:
            Indication of if using single year sampling or not (Avg All Years or Sample 1 Year).
        """
        return self._risk_range

    def get_with_risk_range(self, new_val):
        """Get a copy of this configuration with a new risk year range.

        Args:
            new_val: The new year range to use.

        Returns:
            Copy of this configuration object with the new risk range.
        """
        return Configuration(self._scenario, new_val, self._metric,
                             self._visualization, self._threshold,
                             self._adjustment, self._sig_filter, self._var,
                             self._month, self._loss)

    def get_metric(self):
        """Get the metric which should be displayed in the visualization.

        Returns:
            The metric to display (yield or risk).
        """
        return self._metric

    def get_with_metric(self, new_val):
        """Get a copy of this configuration with a new metric to display to the user.

        Args:
            new_val: The new metric selection.

        Returns:
            Copy of this configuration object with the new metric
        """
        return Configuration(self._scenario, self._risk_range, new_val,
                             self._visualization, self._threshold,
                             self._adjustment, self._sig_filter, self._var,
                             self._month, self._loss)

    def get_visualization(self):
        """Get the visualization type to display to the user.

        Returns:
            The visualization type to display (scatter, neighborhood, etc).
        """
        return self._visualization

    def get_with_visualization(self, new_val):
        """Get a copy of this configuration with a new visualization type selection.

        Args:
            new_val: The new visualization type.

        Returns:
            Copy of this configuration object with a new type of visualization selected.
        """
        return Configuration(self._scenario, self._risk_range, self._metric,
                             new_val, self._threshold, self._adjustment,
                             self._sig_filter, self._var, self._month,
                             self._loss)

    def get_show_acreage(self):
        """Indicate if acreage should be shown.


        Returns:
            True if the circles should be sized according to maize growing acreage or false if they
            should have the same size.
        """
        return self._show_acreage

    def get_viz_type(self):
        """Get the type of visualization to use which may have subtypes.

        Returns:
            Visualization type like map or scatter.
        """
        self._visualization_type

    def get_threshold(self):
        """Get the statistical significance threshold.

        Returns:
            Threshold for significance like p <  0.05.
        """
        return self._threshold

    def get_with_threshold(self, new_val):
        """Get a copy of this configuration with a new statistical significance threshold.

        Args:
            new_val: The new loss threshold option.

        Returns:
            Copy of this configuration object with the new significance threshold.
        """
        return Configuration(self._scenario, self._risk_range, self._metric,
                             self._visualization, new_val, self._adjustment,
                             self._sig_filter, self._var, self._month,
                             self._loss)

    def get_adjustment(self):
        """Get the statistical adjustment to apply.

        Get the statistical adjustment for multiple to comparisons to use in determining
        significance.

        Returns:
            Indication of if only significant results (significant only) or all results (all)
            should be displayed.
        """
        return self._adjustment

    def get_with_adjustment(self, new_val):
        """Get a copy of this configuration with a new significance adjustment.

        Args:
            new_val: The new adjustment option.

        Returns:
            Copy of this configuration object with the new significance adjustment.
        """
        return Configuration(self._scenario, self._risk_range, self._metric,
                             self._visualization, self._threshold, new_val,
                             self._sig_filter, self._var, self._month,
                             self._loss)

    def get_sig_filter(self):
        """Determine if a filter should be applied for significance.

        Returns:
             Indication of if only significant results (significant only) or all results (all)
             should be displayed.
        """
        return self._sig_filter

    def get_with_sig_filter(self, new_val):
        """Get a copy of this configuration with a new significance filter option.

        Args:
            new_val: The new significance filter option.

        Returns:
            Copy of this configuration object with the new significance filter option.
        """
        return Configuration(self._scenario, self._risk_range, self._metric,
                             self._visualization, self._threshold,
                             self._adjustment, new_val, self._var, self._month,
                             self._loss)

    def get_var(self):
        """Get the contextualizing dimension to use in understanding the selected metric.

        Returns:
            The variable to use to contextualize the metric like chirps. May also be 'no var' in
            which case no dimension contextualizes the metric.
        """
        return self._var

    def get_with_var(self, new_val):
        """Get a copy of this configuration with a new contextualizing dimension.

        Args:
            new_val: The new choice for contextualizing dimension or no var.

        Returns:
            Copy of this configuration object with the new dimension selection.
        """
        return Configuration(self._scenario, self._risk_range, self._metric,
                             self._visualization, self._threshold,
                             self._adjustment, self._sig_filter, new_val,
                             self._month, self._loss)

    def get_month(self):
        """Get the month from which the contextualizing dimension values should be shown.

        Returns:
            The string name of the month (like jan) from which to display contextual dimensional
            data. Ignored if no context var selected.
        """
        return self._month

    def get_with_month(self, new_val):
        """Get a copy of this configuration with a new contextualizing dimension month.

        Args:
            new_val: The new month from which contextualizing dimension values should be shown.

        Returns:
            Copy of this configuration object with the new month selection.
        """
        return Configuration(self._scenario, self._risk_range, self._metric,
                             self._visualization, self._threshold,
                             self._adjustment, self._sig_filter, self._var,
                             new_val, self._loss)

    def get_loss(self):
        """Get the loss threshold.

        Returns:
            The loss level epxressed as a level of coverage (75% cov, 85% cov).
        """
        return self._loss

    def get_with_loss(self, new_val):
        """Get a copy of this configuration with a new loss threshold.

        Args:
            new_val: The new choice for loss threshold expressed as a coverage level.

        Returns:
            Copy of this configuration object with the new loss threshold selection.
        """
        return Configuration(self._scenario, self._risk_range, self._metric,
                             self._visualization, self._threshold,
                             self._adjustment, self._sig_filter, self._var,
                             self._month, new_val)

    def _set_viz(self, viz_str):
        self._visualization = viz_str

        if viz_str == 'scatter':
            self._visualization_type = 'scatter'
            self._show_acreage = False
        elif viz_str == 'neighborhood':
            self._visualization_type = 'map'
            self._show_acreage = False
        elif viz_str == 'acreage':
            self._visualization_type = 'map'
            self._show_acreage = True
        else:
            raise RuntimeError('Unknwon viz descsriptor: ' + viz_str)


class ConfigurationPresenter:
    """Meta-widget which allows the user to control a Configuration."""

    def __init__(self, sketch, x, y, initial_config, on_change):
        """Create a new configuration presenter meta-widget made up of more general widgets.

        Args:
            sketch: The sketchingpy sketch in which to create these widgets.
            x: The horizontal coordinate at which the meta-widget should be constructed.
            y: The vertical coordinate at which the meta-widget should be constructed.
            initial_config: Configuration object with the initial values to be shown across this
                widget.
            on_change: Function to call with a new Configuration when changes are made within this
                widget.
        """
        self._sketch = sketch
        self._x = x
        self._y = y
        self._on_change = on_change

        current_y = 0

        self._config = initial_config

        self._viz_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            current_y,
            'Visualization', ['scatter', 'neighborhood', 'acreage'],
            self._config.get_visualization(),
            lambda x: self._set_config(self._config.get_with_visualization(x)),
            keyboard_button='v')

        current_y += self._viz_buttons.get_height() + 11

        self._metric_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            current_y,
            'Metric', ['yield', 'risk'],
            self._config.get_metric(),
            lambda x: self._set_config(self._config.get_with_metric(x)),
            keyboard_button='o')

        current_y += self._metric_buttons.get_height() + 11

        self._loss_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            current_y,
            'Loss', ['75% cov', '85% cov'],
            self._config.get_loss(),
            lambda x: self._set_config(self._config.get_with_loss(x)),
            keyboard_button='c')

        current_y += self._viz_buttons.get_height() + 24

        self._scenario_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            current_y,
            'Scenario', ['2030 series', '2050 series'],
            self._config.get_scenario(),
            lambda x: self._set_config(self._config.get_with_scenario(x)),
            keyboard_button='y')

        current_y += self._scenario_buttons.get_height() + 11

        self._range_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            current_y,
            'Range of Risk', ['Sample 1 Year', 'Avg All Years'],
            self._config.get_risk_range(),
            lambda x: self._set_config(self._config.get_with_risk_range(x)),
            keyboard_button='s')

        current_y += self._range_buttons.get_height() + 24

        self._threshold_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            current_y,
            'Threshold', ['p <  0.05', 'p <  0.10'],
            self._config.get_threshold(),
            lambda x: self._set_config(self._config.get_with_threshold(x)),
            keyboard_button='t')

        current_y += self._threshold_buttons.get_height() + 11

        self._adj_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            current_y,
            'Adjustment', ['Bonferroni', 'no correction'],
            self._config.get_adjustment(),
            lambda x: self._set_config(self._config.get_with_adjustment(x)),
            keyboard_button='b')

        current_y += self._adj_buttons.get_height() + 11

        self._filter_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            current_y,
            'Filter', ['significant only', 'all'],
            self._config.get_sig_filter(),
            lambda x: self._set_config(self._config.get_with_sig_filter(x)),
            keyboard_button='f')

        current_y += self._filter_buttons.get_height() + 24

        self._var_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            current_y,
            'Variable', [
                'no var', 'chirps', 'rhn', 'svp', 'tmax', 'tmin', 'vpd',
                'wbgtmax'
            ],
            self._config.get_var(),
            lambda x: self._set_config(self._config.get_with_var(x)),
            keyboard_button='g')

        self._month_buttons = buttons.ToggleButtonSet(
            self._sketch,
            5,
            const.HEIGHT - const.BUTTON_HEIGHT - 5,
            'Month', [
                'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep',
                'oct', 'nov', 'dec'
            ],
            self._config.get_month(),
            lambda x: self._set_config(self._config.get_with_month(x)),
            make_rows=False,
            narrow=True,
            keyboard_button='m')

    def step(self, mouse_x, mouse_y, clicked, keypress):
        """Update and draw this meta-widget and its sub-widgets.

        Args:
            mouse_x: The horizontal coordinate of the mouse.
            mouse_y: The vertical coordinate of the mouse.
            clicked: Flag indicating if a click has happened since step was last called. True if
                clicked and false otherwise. Also true if tapped.
            keypress: The string key pressed since the last time step was called or None if no
                key pressed.
        """
        self._sketch.push_transform()
        self._sketch.push_style()

        if self._config.get_var() != 'no var':
            self._month_buttons.step(mouse_x, mouse_y, clicked, keypress)

        self._sketch.translate(self._x, self._y)

        mouse_x = mouse_x - self._x
        mouse_y = mouse_y - self._y

        self._scenario_buttons.step(mouse_x, mouse_y, clicked, keypress)
        self._range_buttons.step(mouse_x, mouse_y, clicked, keypress)
        self._metric_buttons.step(mouse_x, mouse_y, clicked, keypress)
        self._viz_buttons.step(mouse_x, mouse_y, clicked, keypress)
        self._threshold_buttons.step(mouse_x, mouse_y, clicked, keypress)
        self._adj_buttons.step(mouse_x, mouse_y, clicked, keypress)
        self._filter_buttons.step(mouse_x, mouse_y, clicked, keypress)
        self._var_buttons.step(mouse_x, mouse_y, clicked, keypress)

        if self._config.get_metric() != 'yield':
            self._loss_buttons.step(mouse_x, mouse_y, clicked, keypress)

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _set_config(self, new_config):
        """Internal callback for when the user changes the Configuration.

        Args:
            new_config: The new Configuration after applying the user's change.
        """
        self._config = new_config
        self._on_change(self._config)
