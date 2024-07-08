import const
import symbols


class LegendPresenter:

    def __init__(self, sketch, x, y, legend_width, legend_height, initial_percents,
        metric, var, viz):
        self._sketch = sketch
        self._x = x
        self._y = y
        self._legend_width = legend_width
        self._legend_height = legend_height
        self._percents = initial_percents
        self._metric = metric
        self._var = var
        self._viz = viz

    def update_data(self, new_data, metric, var, viz):
        self._percents = new_data
        self._metric = metric
        self._var = var
        self._viz = viz

    def step(self, mouse_x, mouse_y, clicked):
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.translate(self._x, self._y)

        self._sketch.set_fill(const.PANEL_BG_COLOR)
        self._sketch.set_stroke(const.ACTIVE_BORDER)
        self._sketch.set_stroke_weight(1)
        self._sketch.set_rect_mode('corner')

        self._sketch.draw_rect(0, 0, self._legend_width, self._legend_height)

        self._sketch.set_text_font(const.FONT_SRC, 11)
        self._sketch.set_text_align('left', 'baseline')
        self._sketch.set_ellipse_mode('radius')

        use_symbols = self._var != 'no var' and self._viz == 'map'

        def draw_series_no_var(x, y, fill_color, category):
            if name == 'not significant':
                self._sketch.set_stroke(fill_color + 'C0')
                self._sketch.set_stroke_weight(2)
                self._sketch.clear_fill()
                self._sketch.draw_ellipse(x, y, 4, 4)
            else:
                self._sketch.clear_stroke()
                self._sketch.set_fill(fill_color)
                self._sketch.draw_ellipse(x, y, 4, 4)

        def draw_series_var(x, y, fill_color, category):
            strategy = symbols.get_strategy(self._sketch, category)
            strategy(x, y, fill_color)

        if not use_symbols:
            draw_series = draw_series_no_var
        else:
            draw_series = draw_series_var

        y = 15
        for percent_record in self._percents:
            name = percent_record.get_name()
            percent = percent_record.get_percent()

            if not use_symbols:
                fill_color = const.CATEGORY_COLORS[name]
            else:
                fill_color = '#A0A0A0'

            draw_series(8, y - 7, fill_color, name)

            self._sketch.clear_stroke()
            self._sketch.set_fill(const.INACTIVE_TEXT_COLOR)
            percent_rounded = round(percent * 100)
            self._sketch.draw_text(17, y - 2, '%d%% see %s' % (percent_rounded, name))

            if not use_symbols:
                self._sketch.set_fill(const.EMBEDDED_BAR_COLOR)
                self._sketch.draw_rect(
                    17,
                    y,
                    self._get_width(percent),
                    5
                )

            y += 15 if use_symbols else 25

        if use_symbols:
            min_val = const.VAR_MINS[self._var]
            max_val = const.VAR_MAXS[self._var]
            extent_val = max([abs(min_val), abs(max_val)])
            self._sketch.set_rect_mode('corner')
            all_colors = list(reversed(const.MAP_SCALE_NEGATIVE)) + const.MAP_SCALE_POSITIVE
            i = 0
            for x in range(38, 38 + 7 * 12, 12):
                self._sketch.clear_stroke()
                color = all_colors[i]
                self._sketch.set_fill(color)
                self._sketch.draw_rect(x, y, 10, 10)
                i += 1

            self._sketch.set_fill('#A0A0A0')
            self._sketch.set_text_font(const.FONT_SRC, 11)
            self._sketch.set_text_align('right', 'center')
            self._sketch.draw_text(37, y + 5, '-%.1f' % extent_val)
            self._sketch.set_text_align('left', 'center')
            self._sketch.draw_text(38 + 7 * 12 + 1, y + 5, '+%.1f' % extent_val)

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _get_width(self, percent):
        working_width = self._legend_width - 40
        return percent * working_width
