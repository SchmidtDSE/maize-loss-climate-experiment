"""Visualization of the model sweep.

Visualization of the model sweep which is called the hyperparameters visualization in the
manuscript.

License:
    BSD
"""

import sketchingpy

import buttons
import const


class SweepMainPresenter:
    """Presenter for the sweep visualization."""

    def __init__(self, target, loading_id):
        """Create a new sweep visualization.

        Args:
            target: The ID at which create the visualization or the window title if not rendering
                on web.
            loading_id: The ID of the loading indictaor to hide after initialization.
        """
        self._click_waiting = False
        self._key_waiting = None
        self._last_x = None
        self._last_y = None
        self._requires_refresh = False

        self._sketch = sketchingpy.Sketch2D(const.WIDTH, const.HEIGHT, target, loading_id)

        data = load_data(self._sketch)

        self._scatter_presenter = ScatterPresenter(
            self._sketch,
            5,
            5,
            const.WIDTH - 280,
            const.HEIGHT - 10,
            data,
            lambda: self._request_redraw()
        )
        self._config_presenter = ConfigPresenter(
            self._sketch,
            const.WIDTH - 260,
            5,
            lambda config: self._update_config(config),
            lambda: self._show_all()
        )

        mouse = self._sketch.get_mouse()

        def set_mouse_clicked(mouse):
            self._click_waiting = True

        mouse.on_button_press(set_mouse_clicked)

        keyboard = self._sketch.get_keyboard()

        def set_key_waiting(button):
            self._key_waiting = button.get_name()

        keyboard.on_key_press(set_key_waiting)

        self._sketch.set_fps(10)
        self._sketch.on_step(lambda x: self._draw())
        self._sketch.show()

    def _request_redraw(self):
        """Require that the visualization be redrawn on the next step."""
        self._requires_refresh = True

    def _update_config(self, config):
        """Update the configuration of the tool.

        Args:
            config: New FilterConfig.
        """
        self._scatter_presenter.show_config(config)

    def _show_all(self):
        """Show all options found within the sweep."""
        self._scatter_presenter.show_all()

    def _draw(self):
        """Update this visualization and redraw if needed or requested."""
        mouse = self._sketch.get_mouse()
        mouse_x = mouse.get_pointer_x()
        mouse_y = mouse.get_pointer_y()

        requires_draw_factors = [
            self._requires_refresh,
            mouse_x != self._last_x,
            mouse_y != self._last_y,
            self._click_waiting,
            self._key_waiting is not None,
            self._scatter_presenter.get_awaiting_draw()
        ]
        requires_draw_factors_valid = filter(
            lambda x: x is True,
            requires_draw_factors
        )
        redraw_reasons_count = sum(map(
            lambda x: 1,
            requires_draw_factors_valid
        ))
        requires_draw = redraw_reasons_count > 0

        if not requires_draw:
            return

        self._last_x = mouse_x
        self._last_y = mouse_y
        self._requires_refresh = False

        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.clear(const.BACKGROUND_COLOR)
        self._scatter_presenter.draw(mouse_x, mouse_y, self._click_waiting)
        self._config_presenter.draw(mouse_x, mouse_y, self._click_waiting, self._key_waiting)

        self._sketch.pop_style()
        self._sketch.pop_transform()

        if self._click_waiting:
            self._click_waiting = False
            self._requires_refresh = True

        if self._key_waiting is not None:
            self._key_waiting = None
            self._requires_refresh = True


class FilterConfig:
    """Object representing a configuration of the sweep visualization."""

    def __init__(self, layers, l2, dropout, data_filter):
        """Create a new configuration describing a filter on the sweep results.

        Args:
            layers: The number of layers to consider.
            l2: The L2 regularization level (0 - 1) to consider.
            dropout: The dropout rate (0 - 1) to consider.
            data_filter: The data attribute hidden from training or 'all attrs' if all data
                included.
        """
        self._layers = layers
        self._l2 = l2
        self._dropout = dropout
        self._data_filter = data_filter

    def get_layers(self):
        """Get the number of layers that the user is filtering for.

        Returns:
            The number of layers to consider.
        """
        return self._layers

    def get_with_layers(self, new_layers):
        """Create a copy of this configuration object with a new number of layers.

        Args:
            new_layers: The new value to use.

        Returns:
            Copy of this object but with the new value.
        """
        return FilterConfig(new_layers, self._l2, self._dropout, self._data_filter)

    def get_l2(self):
        """Get the L2 regularization strength that the user is filtering for.

        Returns:
            The L2 regularization level (0 - 1) to consider.
        """
        return self._l2

    def get_with_l2(self, new_l2):
        """Create a copy of this configuration object with a new L2 regularization strength.

        Args:
            new_l2: The new value to use.

        Returns:
            Copy of this object but with the new value.
        """
        return FilterConfig(self._layers, new_l2, self._dropout, self._data_filter)

    def get_dropout(self):
        """Get the dropout rate that the user is filtering for.

        Returns:
            The dropout rate (0 - 1) to consider.
        """
        return self._dropout

    def get_with_dropout(self, new_dropout):
        """Create a copy of this configuration object with a new dropout rate.

        Args:
            new_dropout: The new dropout rate to use.

        Returns:
            Copy of this object but with the new value.
        """
        return FilterConfig(self._layers, self._l2, new_dropout, self._data_filter)

    def get_data_filter(self):
        """Get the data attribute filter that the user has applied.

        Returns:
            The data attribute hidden from training or 'all attrs' if all data included.
        """
        return self._data_filter

    def get_with_data_filter(self, new_filter):
        """Create a copy of this configuration object with a new data filter.

        Args:
            new_filter: The new value to use.

        Returns:
            Copy of this object but with the new value.
        """
        return FilterConfig(self._layers, self._l2, self._dropout, new_filter)


class SweepResult:
    """Object representing a single model candidate within the sweep."""

    def __init__(self, block, layers, l2, dropout, mean_error, std_err,
        train_mean_error, train_std_err):
        """Create a new record of a sweep outcome.

        Args:
            block: The variable name that was blocked from training or 'all attrs' if no blocks.
            layers: The number of hidden layers used in the candidate.
            l2: The L2 regularization strength used in the candidate.
            dropout: The dropout rate used in this candidate.
            mean_error: The validation set MAE when predicting mean found for this candidate.
            std_err: The validation set MAE when predicting std found for this candidate.
            train_mean_error: The training set MAE when predicting mean found for this candidate.
            train_std_err: The training set MAE when predicting mean found for this candidate.
        """
        self._block = block
        self._layers = layers
        self._l2 = l2
        self._dropout = dropout
        self._mean_error = mean_error
        self._std_err = std_err
        self._train_mean_error = train_mean_error
        self._train_std_err = train_std_err

    def get_block(self):
        """Get the variable blocked from training.

        Returns:
            The variable name that was blocked from training or 'all attrs' if no blocks.
        """
        return self._block

    def get_layers(self):
        """Get the numer of hidden layers in the model candidate."""
        return self._layers

    def get_l2(self):
        """Get the L2 strength that was attempted in this candidate."""
        return self._l2

    def get_dropout(self):
        """Get the dropout rate that was attempted in this candidate."""
        return self._dropout

    def get_mean_error(self):
        """Get the MAE in validation set for mean prediction."""
        return self._mean_error

    def get_std_err(self):
        """Get the MAE in validation set for standard deviation prediction."""
        return self._std_err

    def get_train_mean_error(self):
        """Get the MAE in training set for mean prediction."""
        return self._train_mean_error

    def get_train_std_err(self):
        """Get the MAE in training set for standard deviation prediction."""
        return self._train_std_err

    def matches_filter(self, filter_config):
        """Determine if this candidate should be included in a results set for a filter.

        Args:
            filter_config: The FitlerConfig for which we are testing inclusion.

        Returns:
            True if should be included and false otherwise.
        """

        def floats_match(a, b):
            return abs(a - b) < 0.0001

        block_matches = filter_config.get_data_filter() == self.get_block()
        layers_match = filter_config.get_layers() == self.get_layers()
        l2_match = floats_match(filter_config.get_l2(), self.get_l2())
        dropout_match = floats_match(filter_config.get_dropout(), self.get_dropout())
        matches = [block_matches, layers_match, l2_match, dropout_match]
        unmatched = filter(lambda x: x is False, matches)
        count_unmatched = sum(map(lambda x: 1, unmatched))
        return count_unmatched == 0


def parse_record(raw_record):
    """Parse a sweep output dataset.

    Args:
        raw_record: The raw record as a primitives only dictionary.

    Returns:
        SweepResult after parsing values into exepcted types.
    """
    return SweepResult(
        raw_record['block'],
        int(raw_record['layers']),
        float(raw_record['l2Reg']),
        float(raw_record['dropout']),
        float(raw_record['validMean']),
        float(raw_record['validStd']),
        float(raw_record['trainMean']),
        float(raw_record['trainStd'])
    )


def load_data(sketch, loc='data/sweep_ag_all.csv'):
    """Load all available sweep data.

    Args:
        loc: The location (path) as string from which to load data.

    Returns:
        list of SweepResult.
    """
    data = sketch.get_data_layer().get_csv(loc)
    data_in_scope = filter(lambda x: x['allowCount'] == '1', data)
    return [parse_record(x) for x in data_in_scope]


class ConfigPresenter:
    """Presenter which allows the user to change the sweep filter."""

    def __init__(self, sketch, x, y, on_config_change, on_run_sweep):
        """Create a new configuration presenter / meta-widget.

        Args:
            sketch: The Sketchingpy sketch in which to bulid this widget.
            x: The horizontal position at which to build this widget in pixels.
            y: The horizontal position at which to build this widget in pixels.
            on_config_change: Callback to invoke with a FilterConfig when the filter settings are
                changed by the user.
            on_run_sweep: Callback to invoke when the user requests a sweep.
        """
        self._sketch = sketch
        self._x = x
        self._y = y
        self._on_config_change = on_config_change
        self._on_run_sweep = on_run_sweep

        y = 50
        self._layers_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            y,
            'Layers',
            ['1 layer', '2 layers', '3 layers', '4 layers', '5 layers', '6 layers'],
            '3 layers',
            lambda x: self._change_layers(x),
            keyboard_button='n'
        )

        y += self._layers_buttons.get_height() + 30
        self._l2_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            y,
            'L2',
            [
                'No L2',
                '0.05',
                '0.10',
                '0.15',
                '0.20'
            ],
            'No L2',
            lambda x: self._change_l2(x),
            keyboard_button='l'
        )

        y += self._l2_buttons.get_height() + 30
        self._drop_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            y,
            'Dropout',
            [
                'No Dropout',
                '0.01',
                '0.05',
                '0.10',
                '0.50'
            ],
            'No Dropout',
            lambda x: self._change_dropout(x),
            keyboard_button='d'
        )

        y += self._drop_buttons.get_height() + 30
        self._data_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            y,
            'Data',
            [
                'All Data',
                'no year',
                'no rhn',
                'no rhx',
                'no tmax',
                'no tmin',
                'no chirps',
                'no svp',
                'no vpt',
                'no wbgt'
            ],
            'All Data',
            lambda x: self._change_data_filter(x),
            keyboard_button='b'
        )

        self._filter_config = FilterConfig(
            3,
            0,
            0,
            'all attrs'
        )

        y += self._data_buttons.get_height() + 60
        self._attempt_button = buttons.Button(
            self._sketch,
            260 / 2 - const.BUTTON_WIDTH / 2,
            y,
            'Try Model >>',
            lambda x: self._try_model(),
            keyboard_button='t'
        )

        y += const.BUTTON_HEIGHT + 10
        self._sweep_button = buttons.Button(
            self._sketch,
            260 / 2 - const.BUTTON_WIDTH / 2,
            y,
            'Run Sweep >>',
            lambda x: self._run_sweep(),
            keyboard_button='s'
        )

    def draw(self, mouse_x, mouse_y, click_waiting, keypress):
        """Update and draw this visualization.

        Args:
            mouse_x: The horizontal position of cursor.
            mouse_y: The vertical position of cursor.
            click_waiting: Flag indicating if the mouse button was pressed since the last time draw
                was invoked. True if the mouse button was pressed and false otherwise.
            keypress: String indicating the keyboard key pressed since draw was last called or None
                if no keys pressed since draw last called.
        """
        self._sketch.push_transform()
        self._sketch.push_style()

        mouse_x = mouse_x - self._x
        mouse_y = mouse_y - self._y

        self._sketch.translate(self._x, self._y)
        self._layers_buttons.step(mouse_x, mouse_y, click_waiting, keypress)
        self._l2_buttons.step(mouse_x, mouse_y, click_waiting, keypress)
        self._drop_buttons.step(mouse_x, mouse_y, click_waiting, keypress)
        self._data_buttons.step(mouse_x, mouse_y, click_waiting, keypress)
        self._attempt_button.step(mouse_x, mouse_y, click_waiting, keypress)
        self._sweep_button.step(mouse_x, mouse_y, click_waiting, keypress)

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _change_layers(self, new_val):
        """Internal callback for when the filter's number of hidden layers has changed.

        Args:
            new_val: The new value for this parameter.
        """
        interpreted = int(new_val.split(' ')[0])
        self._filter_config = self._filter_config.get_with_layers(interpreted)

    def _change_l2(self, new_val):
        """Internal callback for when the filter's L2 strength has changed.

        Args:
            new_val: The new value for this parameter.
        """
        interpreted = 0 if new_val == 'No L2' else float(new_val)
        self._filter_config = self._filter_config.get_with_l2(interpreted)

    def _change_dropout(self, new_val):
        """Internal callback for when the filter's dropout rate has changed.

        Args:
            new_val: The new value for this parameter.
        """
        interpreted = 0 if new_val == 'No Dropout' else float(new_val)
        self._filter_config = self._filter_config.get_with_dropout(interpreted)

    def _change_data_filter(self, new_val):
        """Internal callback for when the filter's attribute inclusion has changed.

        Args:
            new_val: The new value for this parameter.
        """
        if new_val == 'All Data':
            interpreted = 'all attrs'
        else:
            interpreted = new_val.split(' ')[-1]
            if interpreted == 'wbgt':
                interpreted = 'wbgtmax'

        self._filter_config = self._filter_config.get_with_data_filter(interpreted)

    def _try_model(self):
        """Use the current filter to find a simulation result."""
        self._on_config_change(self._filter_config)

    def _run_sweep(self):
        """Run a sweep in which all filters are included and a preferred config selected."""
        self._layers_buttons.set_value('6 layers')
        self._l2_buttons.set_value('0.05')
        self._drop_buttons.set_value('0.05')
        self._data_buttons.set_value('All Data')
        self._filter_config = FilterConfig(
            6,
            0.05,
            0.05,
            'all attrs'
        )
        self._on_run_sweep()
        self._try_model()


class ScatterPresenter:
    """Presenter which runs a scatterplot within the sweep visualization."""

    def __init__(self, sketch, x, y, width, height, data, request_draw):
        """Create a new scatterplot.

        Args:
            sketch: The Sketchingpy sketch in which to build the chart.
            x: The horizontal position at which to build the chart.
            y: The vertical position at which to build the chart.
            width: The horizontal size of the chart in pixels.
            height: The vertical size of the chart in pixels.
            data: The SweepResults to display.
            request_draw: Function to call to force a full tool redraw.
        """
        self._sketch = sketch
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._data = data
        self._requires_refresh = True
        self._points = []
        self._request_draw = request_draw

    def get_awaiting_draw(self):
        """Determine if this component requires the full visualization to redraw."""
        return self._requires_refresh

    def draw(self, mouse_x, mouse_y, click_waiting):
        """Update and redraw this component.

        Args:
            mouse_x: The horizontal position of the cursor.
            mouse_y: The vertical position of the cursor.
            click_waiting: Flag indicating if the mouse button was pressed since the last time draw
                was invoked. True if the mouse button was pressed and false otherwise.
        """
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.translate(self._x, self._y)
        self._draw_frame()
        self._draw_horiz_axis()
        self._draw_vert_axis()
        self._draw_contents()

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def show_config(self, config):
        """Show a single model candidate.

        Args:
            config: The FilterConfig for which a model should be displayed.
        """
        matching = list(filter(lambda x: x.matches_filter(config), self._data))
        if len(matching) == 0:
            return

        matched = matching[0]
        self._points.append(self._convert_point(matched))

        self._request_draw()
        self._requires_refresh = True

    def show_all(self):
        """Execute a sweep and show all model candidates."""
        converted_points = map(
            lambda x: self._convert_point(x),
            self._data
        )
        self._points = list(converted_points)

        self._request_draw()
        self._requires_refresh = True

    def _convert_point(self, target):
        """Convert a point into a simplified dict representation.

        Args:
            target: The SweepResult to convert to a simplified dictionary.

        Returns:
            Simplified primitives-only dictionary.
        """
        return {
            'mean': target.get_mean_error(),
            'std': target.get_std_err(),
            'trainMean': target.get_train_std_err(),
            'trainStd': target.get_train_std_err(),
        }

    def _draw_contents(self):
        """Draw the contents of the scatter plot."""
        if self._requires_refresh:
            self._sketch.create_buffer('sweep-points', self._width, self._height)
            self._sketch.enter_buffer('sweep-points')

            self._sketch.push_transform()
            self._sketch.push_style()

            self._sketch.clear_stroke()
            self._sketch.set_ellipse_mode('radius')

            for i in range(0, len(self._points)):
                final_point = i == len(self._points) - 1
                point = self._points[i]

                color = '#1f78b4A0' if final_point else '#C0C0C080'

                if final_point:
                    self._sketch.clear_fill()
                    self._sketch.set_stroke(color)
                    self._sketch.set_stroke_weight(1)
                    self._sketch.draw_line(
                        self._get_x(point['mean']),
                        self._get_y(point['std']),
                        self._get_x(point['trainMean']),
                        self._get_y(point['trainStd'])
                    )

                self._sketch.set_fill(color)
                self._sketch.clear_stroke()
                self._sketch.draw_ellipse(
                    self._get_x(point['mean']),
                    self._get_y(point['std']),
                    5,
                    5
                )

                if final_point:
                    self._sketch.set_text_font(const.FONT_SRC, 12)

                    self._sketch.set_text_align('center', 'bottom')
                    self._sketch.draw_text(
                        self._get_x(point['mean']),
                        self._get_y(point['std']) - 10,
                        'Validation'
                    )

                    self._sketch.set_text_align('center', 'top')
                    self._sketch.draw_text(
                        self._get_x(point['trainMean']),
                        self._get_y(point['trainStd']) + 10,
                        'Train'
                    )

            self._sketch.pop_style()
            self._sketch.pop_transform()
            self._sketch.exit_buffer()

            self._requires_refresh = False

        self._sketch.draw_buffer(0, 0, 'sweep-points')

    def _get_x(self, val):
        """Get the horizontal position corresponding to a mean prediction error.

        Args:
            val: The mean absolute error (MAE).

        Returns:
            Horizontal pixels coordinate.
        """
        if val > 0.2:
            val = 0.2
        return val * 100 / 20 * (self._width - 80 - 20) + 80

    def _get_y(self, val):
        """Get the vertical position corresponding to a mean prediction error.

        Args:
            val: The mean absolute error (MAE).

        Returns:
            Vertical pixels coordinate.
        """
        if val > 0.2:
            val = 0.2
        offset = val * 100 / 20 * (self._height - 50 - 20) + 50
        return self._height - offset

    def _draw_horiz_axis(self):
        """Draw a chart axis for error in predicting mean."""
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.clear_stroke()
        self._sketch.set_fill('#333333')
        self._sketch.set_text_font(const.FONT_SRC, 12)
        self._sketch.set_text_align('center', 'top')

        tick_points_int = range(0, 25, 5)
        tick_points_float = map(lambda x: x / 100, tick_points_int)
        for val in tick_points_float:
            self._sketch.draw_text(
                self._get_x(val),
                self._height - 45,
                ('>' if val >= 0.199 else '') + ('%d%%' % round(val * 100))
            )

        self._sketch.set_text_align('center', 'center')
        self._sketch.push_transform()
        self._sketch.translate(
            self._get_x(0.10),
            self._height - 20
        )
        self._sketch.draw_text(
            0,
            0,
            'Error predicting yield distribution mean (% of yield, MAE)'
        )
        self._sketch.pop_transform()

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _draw_vert_axis(self):
        """Draw a chart axis for error in predicting standard deviation."""
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.clear_stroke()
        self._sketch.set_fill('#333333')
        self._sketch.set_text_font(const.FONT_SRC, 12)
        self._sketch.set_text_align('right', 'center')

        tick_points_int = range(0, 25, 5)
        tick_points_float = map(lambda x: x / 100, tick_points_int)
        for val in tick_points_float:
            self._sketch.draw_text(
                48,
                self._get_y(val),
                ('>' if val >= 0.199 else '') + ('%d%%' % round(val * 100))
            )

        self._sketch.set_text_align('center', 'center')
        self._sketch.push_transform()
        self._sketch.set_angle_mode('degrees')
        self._sketch.translate(
            12,
            self._get_y(0.10)
        )
        self._sketch.rotate(-90)
        self._sketch.draw_text(
            0,
            0,
            'Error predicting yield distribution std (% of yield, MAE)'
        )
        self._sketch.pop_transform()

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _draw_frame(self):
        """Draw the background frame for this scatterplot."""
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.set_fill(const.PANEL_BG_COLOR)
        self._sketch.set_stroke(const.INACTIVE_BORDER)
        self._sketch.set_stroke_weight(1)
        self._sketch.draw_rect(0, 0, self._width, self._height)

        self._sketch.pop_style()
        self._sketch.pop_transform()


def main():
    """Entrypoint for this visualization."""
    presenter = SweepMainPresenter('Sweep Viz', None)
    assert presenter is not None


if __name__ == '__main__':
    main()
