import math

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

        self._viz_buttons = ToggleButtonSet(
            self._sketch,
            0,
            current_y,
            'Visualization',
            ['scatter', 'map'],
            self._config.get_visualization(),
            lambda x: self._set_config(self._config.get_with_visualization(x))
        )

        current_y += self._viz_buttons.get_height() + 12

        self._metric_buttons = ToggleButtonSet(
            self._sketch,
            0,
            current_y,
            'Metric',
            ['yield', 'risk'], #['yield', 'risk', 'adaptation'],
            self._config.get_metric(),
            lambda x: self._set_config(self._config.get_with_metric(x))
        )

        current_y += self._metric_buttons.get_height() + 12

        self._loss_buttons = ToggleButtonSet(
            self._sketch,
            0,
            current_y,
            'Loss',
            ['75% cov', '85% cov'],
            self._config.get_loss(),
            lambda x: self._set_config(self._config.get_with_loss(x))
        )

        current_y += self._viz_buttons.get_height() + 25

        self._scenario_buttons = ToggleButtonSet(
            self._sketch,
            0,
            current_y,
            'Scenario',
            ['2030 series', '2050 series'],
            self._config.get_scenario(),
            lambda x: self._set_config(self._config.get_with_scenario(x))
        )

        current_y += self._scenario_buttons.get_height() + 12

        self._range_buttons = ToggleButtonSet(
            self._sketch,
            0,
            current_y,
            'Range of Risk',
            ['Sample 1 Year', 'Avg All Years'],
            self._config.get_risk_range(),
            lambda x: self._set_config(self._config.get_with_risk_range(x))
        )

        current_y += self._range_buttons.get_height() + 25

        self._threshold_buttons = ToggleButtonSet(
            self._sketch,
            0,
            current_y,
            'Threshold',
            ['p <  0.05', 'p <  0.10'],
            self._config.get_threshold(),
            lambda x: self._set_config(self._config.get_with_threshold(x))
        )

        current_y += self._threshold_buttons.get_height() + 12

        self._adj_buttons = ToggleButtonSet(
            self._sketch,
            0,
            current_y,
            'Adjustment',
            ['Bonferroni', 'no correction'],
            self._config.get_adjustment(),
            lambda x: self._set_config(self._config.get_with_adjustment(x))
        )

        current_y += self._adj_buttons.get_height() + 12

        self._filter_buttons = ToggleButtonSet(
            self._sketch,
            0,
            current_y,
            'Filter',
            ['significant only', 'all'],
            self._config.get_sig_filter(),
            lambda x: self._set_config(self._config.get_with_sig_filter(x))
        )

        current_y += self._filter_buttons.get_height() + 25

        self._var_buttons = ToggleButtonSet(
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
            lambda x: self._set_config(self._config.get_with_var(x))
        )

        self._month_buttons = ToggleButtonSet(
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
            narrow=True
        )

    def step(self, mouse_x, mouse_y, clicked):
        self._sketch.push_transform()
        self._sketch.push_style()

        if self._config.get_var() != 'no var':
            self._month_buttons.step(mouse_x, mouse_y, clicked)

        self._sketch.translate(self._x, self._y)

        mouse_x = mouse_x - self._x
        mouse_y = mouse_y - self._y

        self._scenario_buttons.step(mouse_x, mouse_y, clicked)
        self._range_buttons.step(mouse_x, mouse_y, clicked)
        self._metric_buttons.step(mouse_x, mouse_y, clicked)
        self._viz_buttons.step(mouse_x, mouse_y, clicked)
        self._threshold_buttons.step(mouse_x, mouse_y, clicked)
        self._adj_buttons.step(mouse_x, mouse_y, clicked)
        self._filter_buttons.step(mouse_x, mouse_y, clicked)
        self._var_buttons.step(mouse_x, mouse_y, clicked)

        if self._config.get_metric() != 'yield':
            self._loss_buttons.step(mouse_x, mouse_y, clicked)

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _set_config(self, new_config):
        self._config = new_config
        self._on_change(self._config)


class ToggleButtonSet:

    def __init__(self, sketch, x, y, label, options, selected, on_change, make_rows = True,
        narrow = False):
        self._sketch = sketch
        self._x = x
        self._y = y
        self._label = label
        self._options = options
        self._on_change = on_change
        self._selected = selected
        self._make_rows = make_rows
        self._narrow = narrow

    def set_value(self, option):
        self._selected = option

    def step(self, mouse_x, mouse_y, clicked):
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.translate(self._x, self._y)

        self._sketch.set_stroke_weight(1)
        self._sketch.set_rect_mode('corner')
        self._sketch.set_text_font(const.FONT_SRC, 14)
        self._sketch.set_text_align('center', 'baseline')

        button_x = 0
        button_y = 0
        mouse_x = mouse_x - self._x
        mouse_y = mouse_y - self._y

        def get_stroke_color(is_hovering, is_active):
            if is_hovering:
                return const.HOVER_BORDER
            elif is_active:
                return const.ACTIVE_BORDER
            else:
                return const.INACTIVE_BORDER

        def get_text_color(is_hovering, is_active):
            if is_hovering or is_active:
                return const.ACTIVE_TEXT_COLOR
            else:
                return const.INACTIVE_TEXT_COLOR

        if self._narrow:
            button_width = const.BUTTON_WIDTH_NARROW
        else:
            button_width = const.BUTTON_WIDTH_COMPACT if len(self._options) > 2 else const.BUTTON_WIDTH
        
        i = 1
        for option in self._options:
            is_active = self._selected == option
            is_hovering_x = mouse_x > button_x and mouse_x < button_x + button_width
            is_hovering_y = mouse_y > button_y and mouse_y < const.BUTTON_HEIGHT + button_y
            is_hovering = is_hovering_x and is_hovering_y

            self._sketch.set_fill(const.BUTTON_BG_COLOR)
            self._sketch.set_stroke(get_stroke_color(is_hovering, is_active))
            self._sketch.draw_rect(button_x, button_y, button_width, const.BUTTON_HEIGHT)

            self._sketch.clear_stroke()
            self._sketch.set_fill(get_text_color(is_hovering, is_active))
            self._sketch.draw_text(
                button_x + button_width / 2,
                button_y + const.BUTTON_HEIGHT / 2 + 3,
                str(option)
            )

            if is_hovering and clicked:
                self._selected = option
                self._on_change(option)

            if i % 3 == 0 and self._make_rows:
                button_x = 0
                button_y += const.BUTTON_HEIGHT + 5
            else:
                button_x += button_width + 5

            i += 1

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def get_height(self):
        if self._make_rows:
            rows = math.ceil(len(self._options) / 3)
            return const.BUTTON_HEIGHT * rows
        else:
            return const.BUTTON_HEIGHT


class Button:

    def __init__(self, sketch, x, y, label, on_click, narrow=False):
        self._sketch = sketch
        self._x = x
        self._y = y
        self._label = label
        self._on_click = on_click
        self._narrow = narrow

    def step(self, mouse_x, mouse_y, clicked):
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.translate(self._x, self._y)

        self._sketch.set_stroke_weight(1)
        self._sketch.set_rect_mode('corner')
        self._sketch.set_text_font(const.FONT_SRC, 14)
        self._sketch.set_text_align('center', 'baseline')

        button_x = 0
        button_y = 0
        mouse_x = mouse_x - self._x
        mouse_y = mouse_y - self._y

        def get_stroke_color(is_hovering):
            if is_hovering:
                return const.HOVER_BORDER
            else:
                return const.INACTIVE_BORDER

        def get_text_color(is_hovering):
            if is_hovering:
                return const.ACTIVE_TEXT_COLOR
            else:
                return const.INACTIVE_TEXT_COLOR

        if self._narrow:
            button_width = const.BUTTON_WIDTH_NARROW
        else:
            button_width = const.BUTTON_WIDTH

        is_hovering_x = mouse_x > button_x and mouse_x < button_x + button_width
        is_hovering_y = mouse_y > button_y and mouse_y < const.BUTTON_HEIGHT + button_y
        is_hovering = is_hovering_x and is_hovering_y

        self._sketch.set_fill(const.BUTTON_BG_COLOR)
        self._sketch.set_stroke(get_stroke_color(is_hovering))
        self._sketch.draw_rect(button_x, button_y, button_width, const.BUTTON_HEIGHT)

        self._sketch.clear_stroke()
        self._sketch.set_fill(get_text_color(is_hovering))
        self._sketch.draw_text(
            button_x + button_width / 2,
            button_y + const.BUTTON_HEIGHT / 2 + 3,
            str(self._label)
        )

        if is_hovering and clicked:
            self._on_click(self._label)

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def get_height(self):
        return const.BUTTON_HEIGHT
