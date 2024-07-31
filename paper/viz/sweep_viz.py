import sketchingpy

import buttons
import const


class SweepMainPresenter:

    def __init__(self, target, loading_id):
        self._click_waiting = False
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

        self._sketch.set_fps(10)
        self._sketch.on_step(lambda x: self._draw())
        self._sketch.show()

    def _request_redraw(self):
        self._requires_refresh = True

    def _update_config(self, config):
        self._scatter_presenter.show_config(config)

    def _show_all(self):
        self._scatter_presenter.show_all()

    def _draw(self):
        mouse = self._sketch.get_mouse()
        mouse_x = mouse.get_pointer_x()
        mouse_y = mouse.get_pointer_y()

        requires_draw_factors = [
            self._requires_refresh,
            mouse_x != self._last_x,
            mouse_y != self._last_y,
            self._click_waiting,
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
        self._config_presenter.draw(mouse_x, mouse_y, self._click_waiting)

        self._sketch.pop_style()
        self._sketch.pop_transform()

        if self._click_waiting:
            self._click_waiting = False
            self._requires_refresh = True


class FilterConfig:

    def __init__(self, layers, l2, dropout, data_filter):
        self._layers = layers
        self._l2 = l2
        self._dropout = dropout
        self._data_filter = data_filter
    
    def get_layers(self):
        return self._layers

    def get_with_layers(self, new_layers):
        return FilterConfig(new_layers, self._l2, self._dropout, self._data_filter)
    
    def get_l2(self):
        return self._l2

    def get_with_l2(self, new_l2):
        return FilterConfig(self._layers, new_l2, self._dropout, self._data_filter)
    
    def get_dropout(self):
        return self._dropout

    def get_with_dropout(self, new_dropout):
        return FilterConfig(self._layers, self._l2, new_dropout, self._data_filter)
    
    def get_data_filter(self):
        return self._data_filter

    def get_with_data_filter(self, new_filter):
        return FilterConfig(self._layers, self._l2, self._dropout, new_filter)


class SweepResult:

    def __init__(self, block, layers, l2, dropout, mean_error, std_err,
        train_mean_error, train_std_err):
        self._block = block
        self._layers = layers
        self._l2 = l2
        self._dropout = dropout
        self._mean_error = mean_error
        self._std_err = std_err
        self._train_mean_error = train_mean_error
        self._train_std_err = train_std_err
    
    def get_block(self):
        return self._block
    
    def get_layers(self):
        return self._layers
    
    def get_l2(self):
        return self._l2
    
    def get_dropout(self):
        return self._dropout
    
    def get_mean_error(self):
        return self._mean_error
    
    def get_std_err(self):
        return self._std_err

    def get_train_mean_error(self):
        return self._train_mean_error
    
    def get_train_std_err(self):
        return self._train_std_err

    def matches_filter(self, filter_config):

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


def load_data(sketch):
    data = sketch.get_data_layer().get_csv('data/sweep_ag_all.csv')
    data_in_scope = filter(lambda x: x['allowCount'] == '1', data)
    return [parse_record(x) for x in data_in_scope]


class ConfigPresenter:

    def __init__(self, sketch, x, y, on_config_change, on_run_sweep):
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
            lambda x: self._change_layers(x)
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
            lambda x: self._change_l2(x)
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
            lambda x: self._change_dropout(x)
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
            lambda x: self._change_data_filter(x)
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
            lambda x: self._try_model()
        )

        y += const.BUTTON_HEIGHT + 10
        self._sweep_button = buttons.Button(
            self._sketch,
            260 / 2 - const.BUTTON_WIDTH / 2,
            y,
            'Run Sweep >>',
            lambda x: self._run_sweep()
        )

    def draw(self, mouse_x, mouse_y, click_waiting):
        self._sketch.push_transform()
        self._sketch.push_style()

        mouse_x = mouse_x - self._x
        mouse_y = mouse_y - self._y

        self._sketch.translate(self._x, self._y)
        self._layers_buttons.step(mouse_x, mouse_y, click_waiting)
        self._l2_buttons.step(mouse_x, mouse_y, click_waiting)
        self._drop_buttons.step(mouse_x, mouse_y, click_waiting)
        self._data_buttons.step(mouse_x, mouse_y, click_waiting)
        self._attempt_button.step(mouse_x, mouse_y, click_waiting)
        self._sweep_button.step(mouse_x, mouse_y, click_waiting)

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _change_layers(self, new_val):
        interpreted = int(new_val.split(' ')[0])
        self._filter_config = self._filter_config.get_with_layers(interpreted)

    def _change_l2(self, new_val):
        interpreted = 0 if new_val == 'No L2' else float(new_val)
        self._filter_config = self._filter_config.get_with_l2(interpreted)

    def _change_dropout(self, new_val):
        interpreted = 0 if new_val == 'No Dropout' else float(new_val)
        self._filter_config = self._filter_config.get_with_dropout(interpreted)

    def _change_data_filter(self, new_val):
        if new_val == 'All Data':
            interpreted = 'all attrs'
        else:
            interpreted = new_val.split(' ')[-1]
            if interpreted == 'wbgt':
                interpreted = 'wbgtmax'

        self._filter_config = self._filter_config.get_with_data_filter(interpreted)

    def _try_model(self):
        self._on_config_change(self._filter_config)

    def _run_sweep(self):
        self._layers_buttons.set_value('4 layers')
        self._l2_buttons.set_value('0.10')
        self._drop_buttons.set_value('0.01')
        self._data_buttons.set_value('All Data')
        self._filter_config = FilterConfig(
            4,
            0.1,
            0.01,
            'all attrs'
        )
        self._on_run_sweep()
        self._try_model()


class ScatterPresenter:

    def __init__(self, sketch, x, y, width, height, data, request_draw):
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
        return self._requires_refresh

    def draw(self, mouse_x, mouse_y, click_waiting):
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
        matching = list(filter(lambda x: x.matches_filter(config), self._data))
        if len(matching) == 0:
            return

        matched = matching[0]
        self._points.append(self._convert_point(matched))

        self._request_draw()
        self._requires_refresh = True

    def show_all(self):
        converted_points = map(
            lambda x: self._convert_point(x),
            self._data
        )
        self._points = list(converted_points)

        self._request_draw()
        self._requires_refresh = True

    def _convert_point(self, target):
        return {
            'mean': target.get_mean_error(),
            'std': target.get_std_err(),
            'trainMean': target.get_train_std_err(),
            'trainStd': target.get_train_std_err(),
        }

    def _draw_contents(self):
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
                    self._sketch.set_text_font(const.FONT_SRC, 11)
                    
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
        if val > 0.2:
            val = 0.2
        return val * 100 / 20 * (self._width - 80 - 20) + 80

    def _get_y(self, val):
        if val > 0.2:
            val = 0.2
        offset = val * 100 / 20 * (self._height - 50 - 20) + 50
        return self._height - offset

    def _draw_horiz_axis(self):
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.clear_stroke()
        self._sketch.set_fill('#333333')
        self._sketch.set_text_font(const.FONT_SRC, 11)
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
            'Error predicting yield distribution mean (%, MAE)'
        )
        self._sketch.pop_transform()

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _draw_vert_axis(self):
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.clear_stroke()
        self._sketch.set_fill('#333333')
        self._sketch.set_text_font(const.FONT_SRC, 11)
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
            'Error predicting yield distribution std (%, MAE)'
        )
        self._sketch.pop_transform()

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _draw_frame(self):
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.set_fill(const.PANEL_BG_COLOR)
        self._sketch.set_stroke(const.INACTIVE_BORDER)
        self._sketch.set_stroke_weight(1)
        self._sketch.draw_rect(0, 0, self._width, self._height)

        self._sketch.pop_style()
        self._sketch.pop_transform()


def main():
    presenter = SweepMainPresenter('Sweep Viz', None)


if __name__ == '__main__':
    main()
