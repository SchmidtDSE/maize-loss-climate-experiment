import buttons
import const


class Configuration:

    def __init__(self, scenario, risk_range, metric, visualization, threshold, adjustment,
        sig_filter, var, month, loss):
        self._scenario = scenario
        self._risk_range = risk_range
        self._metric = metric
        self._visualization = visualization
        self._threshold = threshold
        self._adjustment = adjustment
        self._sig_filter = sig_filter
        self._var = var
        self._month = month
        self._loss = loss
    
    def get_scenario(self):
        return self._scenario

    def get_with_scenario(self, new_val):
        return Configuration(
            new_val,
            self._risk_range,
            self._metric,
            self._visualization,
            self._threshold,
            self._adjustment,
            self._sig_filter,
            self._var,
            self._month,
            self._loss
        )
    
    def get_risk_range(self):
        return self._risk_range

    def get_with_risk_range(self, new_val):
        return Configuration(
            self._scenario,
            new_val,
            self._metric,
            self._visualization,
            self._threshold,
            self._adjustment,
            self._sig_filter,
            self._var,
            self._month,
            self._loss
        )
    
    def get_metric(self):
        return self._metric

    def get_with_metric(self, new_val):
        return Configuration(
            self._scenario,
            self._risk_range,
            new_val,
            self._visualization,
            self._threshold,
            self._adjustment,
            self._sig_filter,
            self._var,
            self._month,
            self._loss
        )
    
    def get_visualization(self):
        return self._visualization

    def get_with_visualization(self, new_val):
        return Configuration(
            self._scenario,
            self._risk_range,
            self._metric,
            new_val,
            self._threshold,
            self._adjustment,
            self._sig_filter,
            self._var,
            self._month,
            self._loss
        )
    
    def get_threshold(self):
        return self._threshold

    def get_with_threshold(self, new_val):
        return Configuration(
            self._scenario,
            self._risk_range,
            self._metric,
            self._visualization,
            new_val,
            self._adjustment,
            self._sig_filter,
            self._var,
            self._month,
            self._loss
        )
    
    def get_adjustment(self):
        return self._adjustment

    def get_with_adjustment(self, new_val):
        return Configuration(
            self._scenario,
            self._risk_range,
            self._metric,
            self._visualization,
            self._threshold,
            new_val,
            self._sig_filter,
            self._var,
            self._month,
            self._loss
        )
    
    def get_sig_filter(self):
        return self._sig_filter

    def get_with_sig_filter(self, new_val):
        return Configuration(
            self._scenario,
            self._risk_range,
            self._metric,
            self._visualization,
            self._threshold,
            self._adjustment,
            new_val,
            self._var,
            self._month,
            self._loss
        )

    def get_var(self):
        return self._var

    def get_with_var(self, new_val):
        return Configuration(
            self._scenario,
            self._risk_range,
            self._metric,
            self._visualization,
            self._threshold,
            self._adjustment,
            self._sig_filter,
            new_val,
            self._month,
            self._loss
        )

    def get_month(self):
        return self._month

    def get_with_month(self, new_val):
        return Configuration(
            self._scenario,
            self._risk_range,
            self._metric,
            self._visualization,
            self._threshold,
            self._adjustment,
            self._sig_filter,
            self._var,
            new_val,
            self._loss
        )

    def get_loss(self):
        return self._loss

    def get_with_loss(self, new_val):
        return Configuration(
            self._scenario,
            self._risk_range,
            self._metric,
            self._visualization,
            self._threshold,
            self._adjustment,
            self._sig_filter,
            self._var,
            self._month,
            new_val
        )


class ConfigurationPresenter:

    def __init__(self, sketch, x, y, initial_config, on_change):
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
            'Visualization',
            ['scatter', 'map'],
            self._config.get_visualization(),
            lambda x: self._set_config(self._config.get_with_visualization(x)),
            keyboard_button='v'
        )

        current_y += self._viz_buttons.get_height() + 12

        self._metric_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            current_y,
            'Metric',
            ['yield', 'risk'], #['yield', 'risk', 'adaptation'],
            self._config.get_metric(),
            lambda x: self._set_config(self._config.get_with_metric(x)),
            keyboard_button='o'
        )

        current_y += self._metric_buttons.get_height() + 12

        self._loss_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            current_y,
            'Loss',
            ['75% cov', '85% cov'],
            self._config.get_loss(),
            lambda x: self._set_config(self._config.get_with_loss(x)),
            keyboard_button='c'
        )

        current_y += self._viz_buttons.get_height() + 25

        self._scenario_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            current_y,
            'Scenario',
            ['2030 series', '2050 series'],
            self._config.get_scenario(),
            lambda x: self._set_config(self._config.get_with_scenario(x)),
            keyboard_button='y'
        )

        current_y += self._scenario_buttons.get_height() + 12

        self._range_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            current_y,
            'Range of Risk',
            ['Sample 1 Year', 'Avg All Years'],
            self._config.get_risk_range(),
            lambda x: self._set_config(self._config.get_with_risk_range(x)),
            keyboard_button='s'
        )

        current_y += self._range_buttons.get_height() + 25

        self._threshold_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            current_y,
            'Threshold',
            ['p <  0.05', 'p <  0.10'],
            self._config.get_threshold(),
            lambda x: self._set_config(self._config.get_with_threshold(x)),
            keyboard_button='t'
        )

        current_y += self._threshold_buttons.get_height() + 12

        self._adj_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            current_y,
            'Adjustment',
            ['Bonferroni', 'no correction'],
            self._config.get_adjustment(),
            lambda x: self._set_config(self._config.get_with_adjustment(x)),
            keyboard_button='b'
        )

        current_y += self._adj_buttons.get_height() + 12

        self._filter_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            current_y,
            'Filter',
            ['significant only', 'all'],
            self._config.get_sig_filter(),
            lambda x: self._set_config(self._config.get_with_sig_filter(x)),
            keyboard_button='f'
        )

        current_y += self._filter_buttons.get_height() + 25

        self._var_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            current_y,
            'Variable',
            [
                'no var',
                'chirps',
                'rhn',
                'svp',
                'tmax',
                'tmin',
                'vpd',
                'wbgtmax'
            ],
            self._config.get_var(),
            lambda x: self._set_config(self._config.get_with_var(x)),
            keyboard_button='g'
        )

        self._month_buttons = buttons.ToggleButtonSet(
            self._sketch,
            5,
            const.HEIGHT - const.BUTTON_HEIGHT - 5,
            'Month',
            [
                'jan',
                'feb',
                'mar',
                'apr',
                'may',
                'jun',
                'jul',
                'aug',
                'sep',
                'oct',
                'nov',
                'dec'
            ],
            self._config.get_month(),
            lambda x: self._set_config(self._config.get_with_month(x)),
            make_rows=False,
            narrow=True,
            keyboard_button='m'
        )

    def step(self, mouse_x, mouse_y, clicked, keypress):
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
        self._config = new_config
        self._on_change(self._config)