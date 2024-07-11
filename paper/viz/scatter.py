import math

import toolz.itertoolz

import const
import data_struct


class ScatterMainPresenter:

    def __init__(self, sketch, x, y, width, height, records, metric, variable, selected_geohashes,
        on_selection):
        self._sketch = sketch
        self._x = x
        self._y = y
        self._width = width
        self._height = height

        self._records = records
        self._metric = metric
        self._variable = variable
        self._selected_geohashes = selected_geohashes
        self._on_selection = on_selection
        
        self._needs_redraw = False
        self._selecting = False
        self._placed_records = []

        self._make_scatter_image(records, metric, variable)

    def start_selecting(self):
        self._selecting = True

    def step(self, mouse_x, mouse_y, clicked):
        self._sketch.push_transform()
        self._sketch.push_style()

        if self._needs_redraw:
            self._make_scatter_image(
                self._records,
                self._metric,
                self._variable
            )
            self._needs_redraw = False

        self._sketch.translate(self._x, self._y)
        mouse_x_offset = mouse_x - self._x
        mouse_y_offset = mouse_y - self._y

        self._sketch.set_fill(const.PANEL_BG_COLOR)
        self._sketch.set_stroke(const.ACTIVE_BORDER)
        self._sketch.set_stroke_weight(1)
        self._sketch.set_rect_mode('corner')

        usable_height = self._get_usable_height()
        self._sketch.draw_rect(0, 0, self._width, usable_height)

        self._sketch.draw_buffer(0, 0, 'scatter')

        hovering_x = mouse_x_offset > 0 and mouse_x_offset < self._width
        hovering_y = mouse_y_offset > 0 and mouse_y_offset < self._height
        hovering = hovering_x and hovering_y
        
        if self._selecting:
            self._sketch.clear_fill()
            self._sketch.set_stroke(const.SELECT_COLOR)
            self._sketch.set_stroke_weight(2)
            self._sketch.set_ellipse_mode('radius')
            
            if hovering:
                self._sketch.draw_ellipse(mouse_x_offset, mouse_y_offset, 20, 20)
            else:
                self._sketch.draw_ellipse(30, 30, 20, 20)

            if clicked:
                self._select_points(mouse_x_offset - 60, mouse_y_offset)
        
        self._sketch.pop_style()
        self._sketch.pop_transform()

    def update_data(self, records, metric, variable, selected_geohashes):
        self._records = records
        self._metric = metric
        self._variable = variable
        self._selected_geohashes = selected_geohashes
        self._needs_redraw = True
        self._selecting = False

    def _select_points(self, rel_x, rel_y):
        in_range = filter(lambda x: x.in_range(rel_x, rel_y, 20), self._placed_records)
        geohashes = set(map(lambda x: x.get_record().get_geohash(), in_range))
        self._on_selection(geohashes)
        self._selecting = False

    def _make_scatter_image(self, records, metric, variable):
        usable_height = self._get_usable_height()
        effective_height = usable_height - 60
        effective_width = self._width - 80
        
        self._sketch.create_buffer('scatter', self._width, usable_height)

        self._sketch.push_transform()
        self._sketch.push_style()
        self._sketch.enter_buffer('scatter')

        self._sketch.translate(60, 0)

        if variable != 'no var':
            metric = metric + 'Var'

        if metric == 'yield':
            min_value_x = const.YIELD_MIN_VALUE
            min_value_y = const.YIELD_MIN_VALUE
            max_value_x = const.YIELD_MAX_VALUE
            max_value_y = const.YIELD_MAX_VALUE
            increment_x = const.YIELD_INCREMENT
            increment_y = const.YIELD_INCREMENT
            format_str_x = lambda x: '%+.0f%%' % (x * 100)
            format_str_y = lambda x: '%+.0f%%' % (x * 100)
            vert_title = 'Change from Yield Expectation (Climate Change)'
            horiz_title = 'Change from Yield Expectation (Counterfactual - No Further Climate Change)'
        elif metric == 'yieldVar':
            min_value_y = const.YIELD_CHANGE_MIN_VALUE
            max_value_y = const.YIELD_CHANGE_MAX_VALUE
            increment_y = const.YIELD_CHANGE_INCREMENT
            format_str_y = lambda x: '%+.0f%%' % (x * 100)
            vert_title = 'Change from Yield Expectation (Climate Change)'
            min_value_x = const.VAR_MINS[variable]
            max_value_x = const.VAR_MAXS[variable]
            increment_x = const.VAR_INCREMENTS[variable]
            horiz_title = 'Mean Change (%s, z)' % variable
            format_str_x = lambda x: '%+.1f' % x
        elif metric == 'risk':
            min_value_y = const.RISK_MIN_VALUE
            max_value_y = const.RISK_MAX_VALUE
            min_value_x = const.YIELD_CHANGE_MIN_VALUE
            max_value_x = const.YIELD_CHANGE_MAX_VALUE
            increment_y = const.RISK_INCREMENT
            increment_x = const.YIELD_CHANGE_INCREMENT
            format_str_x = lambda x: '%+.0f%%' % (x * 100)
            format_str_y = lambda x: '%+.0f%%' % x
            vert_title = 'Change in Claims Rate'
            horiz_title = 'Change from Yield Expectation (Climate Change)'
        elif metric == 'riskVar':
            min_value_y = const.RISK_MIN_VALUE
            max_value_y = const.RISK_MAX_VALUE
            increment_y = const.RISK_INCREMENT
            format_str_y = lambda x: '%+.0f%%' % x
            vert_title = 'Change in Claims Rate'
            min_value_x = const.VAR_MINS[variable]
            max_value_x = const.VAR_MAXS[variable]
            increment_x = const.VAR_INCREMENTS[variable]
            horiz_title = 'Mean Change (%s, z)' % variable
            format_str_x = lambda x: '%+.1f' % x
        elif metric == 'adaptation':
            min_value_x = const.ADAPT_MIN_VALUE
            min_value_y = const.ADAPT_MIN_VALUE
            max_value_x = const.ADAPT_MAX_VALUE
            max_value_y = const.ADAPT_MAX_VALUE
            increment_x = const.ADAPT_INCREMENT
            increment_y = const.ADAPT_INCREMENT
            format_str_x = lambda x: '%+.0f%%' % x
            format_str_y = lambda x: '%+.0f%%' % x
            vert_title = 'Change in Claims Rate'
            horiz_title = 'Adaptation Effect'
        elif metric == 'adaptationVar':
            min_value_y = const.ADAPT_MIN_VALUE
            max_value_y = const.ADAPT_MAX_VALUE
            increment_y = const.ADAPT_INCREMENT
            format_str_y = lambda x: '%+.0f%%' % x
            vert_title = 'Change Catastrophic Probabilty with Adapt'
            min_value_x = const.VAR_MINS[variable]
            max_value_x = const.VAR_MAXS[variable]
            increment_x = const.VAR_INCREMENTS[variable]
            horiz_title = 'Mean Change (%s, z)' % variable
            format_str_x = lambda x: '%+.1f' % x
        else:
            raise RuntimeError('Unknown metric ' + metric)
        
        max_count = const.MAX_COUNT
        total = sum(map(lambda x: x.get_count(), records))

        def get_x_pos(x):
            return (x - min_value_x) / (max_value_x - min_value_x) * effective_width

        def get_y_pos(y):
            raw_value = (y - min_value_y) / (max_value_y - min_value_y) * effective_height
            offset = raw_value + 50
            return usable_height - offset

        def get_radius(r):
            result = math.sqrt(r / max_count * 1000)
            if result < 2:
                return 2
            else:
                return result

        def draw_reference():
            self._sketch.push_transform()
            self._sketch.push_style()

            self._sketch.set_stroke(const.REFERENCE_COLOR)
            self._sketch.clear_fill()

            if metric == 'yield':
                self._sketch.draw_line(
                    get_x_pos(min_value_x),
                    get_y_pos(min_value_y),
                    get_x_pos(max_value_x),
                    get_y_pos(max_value_y)
                )
            else:
                self._sketch.draw_line(
                    get_x_pos(min_value_x),
                    get_y_pos(0),
                    get_x_pos(max_value_x),
                    get_y_pos(0)
                )
                self._sketch.draw_line(
                    get_x_pos(0),
                    get_y_pos(min_value_y),
                    get_x_pos(0),
                    get_y_pos(max_value_y)
                )

            self._sketch.pop_style()
            self._sketch.pop_transform()

        def draw_points():
            self._sketch.push_transform()
            self._sketch.push_style()

            self._sketch.set_ellipse_mode('radius')

            def draw_record(record, is_highlighted):
                if record.get_category() == 'not significant':
                    self._sketch.set_stroke_weight(2)
                    self._sketch.set_stroke(const.CATEGORY_COLORS[record.get_category()] + 'A0')
                    self._sketch.clear_fill()
                else:
                    self._sketch.clear_stroke()
                    self._sketch.set_fill(const.CATEGORY_COLORS[record.get_category()] + 'A0')

                if is_highlighted:
                    self._sketch.set_stroke_weight(1)
                    self._sketch.set_stroke(const.SELECTED_COLOR)
                
                x = get_x_pos(record.get_x_value())
                y = get_y_pos(record.get_y_value())
                r = get_radius(record.get_count() / total)
                self._sketch.draw_ellipse(x, y, r, r)

                self._placed_records.append(data_struct.PlacedRecord(x, y, record))

            for record in filter(lambda x: x.get_geohash() not in self._selected_geohashes, records):
                draw_record(record, False)

            for record in filter(lambda x: x.get_geohash() in self._selected_geohashes, records):
                draw_record(record, True)

            self._sketch.pop_style()
            self._sketch.pop_transform()

        def get_counts(value_getter):
            identity_tuples = map(lambda x: (value_getter(x), x.get_count()), records)
            reduced = toolz.itertoolz.reduceby(
                lambda x: x[0],
                lambda a, b: (a[0], a[1] + b[1]),
                identity_tuples
            )
            return dict(reduced.values())

        def get_hist_size(percent):
            return percent * 25

        def draw_horiz_axis():
            self._sketch.push_transform()
            self._sketch.push_style()

            self._sketch.set_text_font(const.FONT_SRC, 11)
            self._sketch.set_text_align('center', 'top')
            self._sketch.clear_stroke()
            self._sketch.set_rect_mode('corner')

            counts = get_counts(
                lambda x: '%.1f' % (round(x.get_x_value() / increment_x) * increment_x)
            )
            total = sum(counts.values())

            current_value = min_value_x
            while current_value <= max_value_x:
                if abs(current_value - 0) < 0.00001:
                    current_value = 0
                
                self._sketch.set_fill(const.INACTIVE_TEXT_COLOR)
                x_pos = get_x_pos(current_value)
                self._sketch.draw_text(
                    x_pos,
                    effective_height + 12,
                    format_str_x(current_value)
                )
                
                self._sketch.set_fill(const.EMBEDDED_BAR_COLOR)

                if total == 0:
                    height = 0
                else:
                    height = get_hist_size(counts.get('%.1f' % current_value, 0) / total)
                
                if height > 0.1:
                    self._sketch.draw_rect(
                        x_pos - 5,
                        effective_height + 25,
                        10,
                        height
                    )
                
                current_value += increment_x

            self._sketch.set_fill(const.EMBEDDED_BAR_COLOR_TEXT)
            self._sketch.set_text_align('right', 'center')
            self._sketch.draw_text(
                get_x_pos(min_value_x),
                effective_height + 24 + get_hist_size(1),
                '100%'
            )

            self._sketch.set_fill(const.INACTIVE_TEXT_COLOR)
            self._sketch.set_text_align('center', 'top')
            self._sketch.draw_text(
                effective_width / 2,
                effective_height + 45,
                horiz_title
            )

            self._sketch.pop_style()
            self._sketch.pop_transform()

        def draw_vert_axis():
            self._sketch.push_transform()
            self._sketch.push_style()

            self._sketch.set_text_font(const.FONT_SRC, 11)
            self._sketch.set_text_align('right', 'center')
            self._sketch.clear_stroke()
            self._sketch.set_rect_mode('corner')

            counts = get_counts(
                lambda x: '%.1f' % (round(x.get_y_value() / increment_y) * increment_y)
            )
            total = sum(counts.values())

            current_value = min_value_y
            while current_value <= max_value_y:
                if abs(current_value - 0) < 0.00001:
                    current_value = 0

                y_pos = get_y_pos(current_value)
                
                self._sketch.set_fill(const.INACTIVE_TEXT_COLOR)
                self._sketch.draw_text(
                    -1,
                    y_pos,
                    format_str_y(current_value)
                )

                self._sketch.set_fill(const.EMBEDDED_BAR_COLOR)

                if total == 0:
                    width = 0
                else:
                    width = get_hist_size(counts.get('%.1f' % current_value, 0) / total)
                
                if width > 0.1:
                    self._sketch.draw_rect(
                        -1 - width,
                        y_pos + 7,
                        width,
                        10
                    )

                current_value += increment_y

            self._sketch.push_transform()
            self._sketch.set_fill(const.EMBEDDED_BAR_COLOR_TEXT)
            self._sketch.set_text_align('right', 'center')
            self._sketch.translate(
                -get_hist_size(1) - 1,
                get_y_pos(min_value_y)
            )
            self._sketch.rotate(-90)
            self._sketch.draw_text(0, 0, '100%')
            self._sketch.pop_transform()

            self._sketch.push_transform()
            self._sketch.translate(-45, effective_height / 2)
            self._sketch.set_angle_mode('degrees')
            self._sketch.rotate(-90)
            self._sketch.set_fill(const.INACTIVE_TEXT_COLOR)
            self._sketch.set_text_align('center', 'center')
            self._sketch.draw_text(
                0,
                0,
                vert_title
            )
            self._sketch.pop_transform()

            self._sketch.pop_style()
            self._sketch.pop_transform()

        del self._placed_records[:]
        draw_reference()
        draw_points()
        draw_horiz_axis()
        draw_vert_axis()

        self._sketch.pop_style()
        self._sketch.pop_transform()
        self._sketch.exit_buffer()

    def _get_usable_height(self):
        return self._height - 5
