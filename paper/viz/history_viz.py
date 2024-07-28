import copy
import statistics

import sketchingpy

import buttons
import const

HIGH_VARIABILITY_SCENARIO = [200, 140, 240, 190, 270, 145, 270, 170, 290, 20]
LOW_VARIABILITY_SCENARIO =  [190, 200, 190, 195, 200, 210, 200, 205, 220, 160]


class HistoryMainPresenter:

    def __init__(self, target, loading_id):
        self._sketch = sketchingpy.Sketch2D(const.WIDTH, const.HEIGHT, target, loading_id)
        self._sketch.set_fps(10)

        self._click_waiting = False
        
        self._last_mouse_x = None
        self._last_mouse_y = None
        self._change_waiting = False

        self._chart_presenter = HistoryChartPresenter(
            self._sketch,
            5,
            20,
            const.WIDTH - 10,
            const.HEIGHT - 5 - 20 - (const.BUTTON_HEIGHT + 10),
            LOW_VARIABILITY_SCENARIO
        )

        self._threshold_type_buttons = buttons.ToggleButtonSet(
            self._sketch,
            5,
            const.HEIGHT - (const.BUTTON_HEIGHT + 7),
            'Threshold',
            ['Average-based', 'Stdev-based'],
            'Average-based',
            lambda new_val: self._update_threshold(new_val)
        )

        self._scenario_buttons = buttons.ToggleButtonSet(
            self._sketch,
            const.WIDTH - 10 - const.BUTTON_WIDTH * 2,
            const.HEIGHT - (const.BUTTON_HEIGHT + 7),
            'Example',
            ['Low Stability', 'High Stability'],
            'High Stability',
            lambda new_val: self._update_stability(new_val)
        )

        mouse = self._sketch.get_mouse()

        def set_mouse_clicked(mouse):
            self._click_waiting = True

        mouse.on_button_press(set_mouse_clicked)

        self._sketch.on_step(lambda x: self._step())
        self._sketch.show()

    def _step(self):
        mouse = self._sketch.get_mouse()
        mouse_x = mouse.get_pointer_x()
        mouse_y = mouse.get_pointer_y()
        
        mouse_x_same = mouse_x == self._last_mouse_x
        mouse_y_same = mouse_y == self._last_mouse_y
        click_clear = not self._click_waiting
        mouse_same = mouse_x_same and mouse_y_same and click_clear
        if mouse_same and not self._change_waiting:
            return
        else:
            self._last_mouse_x = mouse_x
            self._last_mouse_y = mouse_y
            self._change_waiting = False

        self._sketch.push_transform()
        self._sketch.push_style() 

        self._sketch.clear(const.BACKGROUND_COLOR)

        self._draw_annotation()
        self._chart_presenter.step(mouse_x, mouse_y, self._click_waiting)
        self._threshold_type_buttons.step(mouse_x, mouse_y, self._click_waiting)
        self._scenario_buttons.step(mouse_x, mouse_y, self._click_waiting)

        self._sketch.pop_style()
        self._sketch.pop_transform()

        self._click_waiting = False

    def _draw_annotation(self):
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.set_fill('#333333')
        self._sketch.clear_stroke()
        self._sketch.set_text_font(const.FONT_SRC, 12)
        self._sketch.set_text_align('left', 'bottom')
        self._sketch.draw_text(5, 17, const.HISTORY_INSTRUCTION)

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _update_threshold(self, new_value):
        self._change_waiting = True
        self._chart_presenter.set_use_std(new_value == 'Stdev-based')

    def _update_stability(self, new_setting):
        self._change_waiting = True
        if new_setting == 'Low Stability':
            self._chart_presenter.set_values(HIGH_VARIABILITY_SCENARIO)
        else:
            self._chart_presenter.set_values(LOW_VARIABILITY_SCENARIO)


class HistoryChartPresenter:

    def __init__(self, sketch, x, y, width, height, start_values):
        self._sketch = sketch
        self._use_std = False
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._values = start_values

    def step(self, mouse_x_abs, mouse_y_abs, clicked):
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.translate(self._x, self._y)
        mouse_x = mouse_x_abs - self._x
        mouse_y = mouse_y_abs - self._y

        self._sketch.set_fill(const.PANEL_BG_COLOR)
        self._sketch.set_stroke(const.INACTIVE_BORDER)
        self._sketch.set_stroke_weight(1)
        self._sketch.set_rect_mode('corner')
        self._sketch.draw_rect(0, 0, self._width, self._height)

        self._draw_hover(mouse_x, mouse_y)
        self._draw_candidate(mouse_x, mouse_y, clicked)
        self._draw_x_axis()
        self._draw_y_axis()
        self._draw_average()
        self._draw_content()

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def set_use_std(self, new_use_std):
        self._use_std = new_use_std

    def set_values(self, new_values):
        self._values = new_values

    def _get_year(self, mouse_x, mouse_y):
        if mouse_x < 0 or mouse_y < 0:
            return None
        elif mouse_x > (self._width - 100) or mouse_y > self._height:
            return None
        else:
            for year in range(1, 11):
                year_x = self._get_x(year)
                if abs(mouse_x - year_x) < 30:
                    return year

            return None

    def _draw_hover(self, mouse_x, mouse_y):
        year = self._get_year(mouse_x, mouse_y)
        if year is None:
            return

        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.set_fill(const.HOVER_AREA_COLOR)
        self._sketch.clear_stroke()
        self._sketch.set_rect_mode('corner')

        start_x = self._get_x(year) - 30
        self._sketch.draw_rect(start_x, 5, 60, self._height - 40)

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _draw_x_axis(self):
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.set_text_align('center', 'top')
        self._sketch.set_fill(const.INACTIVE_TEXT_COLOR)
        self._sketch.clear_stroke()
        self._sketch.set_text_font(const.FONT_SRC, 12)

        self._sketch.draw_text(self._get_x(5), self._height - 14, 'Year')

        for year in range(1, 11):
            self._sketch.draw_text(self._get_x(year), self._height - 28, year)

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _draw_y_axis(self):
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.set_text_align('right', 'center')
        self._sketch.set_fill(const.INACTIVE_TEXT_COLOR)
        self._sketch.clear_stroke()
        self._sketch.set_text_font(const.FONT_SRC, 12)

        for value in range(0, 450, 50):
            self._sketch.draw_text(45, self._get_y(value), value)

        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.translate(13, self._get_y(200))
        self._sketch.set_angle_mode('degrees')
        self._sketch.rotate(-90)
        self._sketch.set_text_align('center', 'center')
        self._sketch.draw_text(0, 0, 'Yield')

        self._sketch.pop_style()
        self._sketch.pop_transform()

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _draw_candidate(self, mouse_x, mouse_y, clicked):
        selected_year = self._get_year(mouse_x, mouse_y)
        if selected_year is None:
            return

        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.set_stroke(const.HISTORY_PENDING_COLOR)
        self._sketch.clear_fill()
        self._sketch.set_stroke_weight(2)

        new_value = self._invert_y(mouse_y)
        new_values = copy.deepcopy(self._values)
        new_values[selected_year - 1] = new_value

        shape = self._sketch.start_shape(self._get_x(1), self._get_y(new_values[0]))
        year = 2
        
        for value in new_values[1:]:
            shape.add_line_to(self._get_x(year), self._get_y(value))
            year += 1

        shape.end()
        self._sketch.draw_shape(shape)

        if clicked:
            self._values = new_values

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _draw_content(self):
        self._sketch.push_transform()
        self._sketch.push_style()

        average = statistics.mean(self._values)
        std = statistics.stdev(self._values)

        if self._use_std:
            threshold = -2.1
            format_str = '%s%.1f std'
            get_delta = lambda x: (x - average) / std
        else:
            threshold = -25
            format_str = '%s%.0f%% of avg'
            get_delta = lambda x: (x - average) / average * 100

        def draw_line():
            self._sketch.push_transform()
            self._sketch.push_style()
        
            self._sketch.set_stroke(const.HISTORY_BODY_COLOR)
            self._sketch.clear_fill()
            self._sketch.set_stroke_weight(3)

            shape = self._sketch.start_shape(self._get_x(1), self._get_y(self._values[0]))
            year = 2
            
            for value in self._values[1:]:
                shape.add_line_to(self._get_x(year), self._get_y(value))
                year += 1

            shape.end()
            self._sketch.draw_shape(shape)

            self._sketch.pop_style()
            self._sketch.pop_transform()

        def draw_dots():
            self._sketch.push_transform()
            self._sketch.push_style()
        
            self._sketch.clear_stroke()
            self._sketch.set_ellipse_mode('radius')
            self._sketch.set_text_font(const.FONT_SRC, 12)
            self._sketch.set_text_align('center', 'center')

            year = 1
            for value in self._values:
                x = self._get_x(year)
                y = self._get_y(value)

                text_offset = -17 if value > average else 17
                delta = get_delta(value)
                sign_str = '+' if delta > 0 else ''
                percent_str = format_str % (sign_str, delta)
                
                if delta < threshold:
                    color = const.HISTORY_BODY_LOSS_COLOR
                else:
                    color = const.HISTORY_BODY_COLOR
                
                self._sketch.set_fill(color)
                self._sketch.draw_ellipse(x, y, 7, 7)
                self._sketch.draw_text(x, y + text_offset, percent_str)

                year += 1

            self._sketch.pop_style()
            self._sketch.pop_transform()

        draw_line()
        draw_dots()

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _draw_average(self):
        self._sketch.push_transform()
        self._sketch.push_style()

        average = statistics.mean(self._values)

        if self._use_std:
            std = statistics.stdev(self._values)
            with_loss = filter(lambda x: (x - average) / std < -2.1, self._values)
        else:
            with_loss = filter(lambda x: (x - average) / average < -0.25, self._values)
        
        losses = sum(map(lambda x: 1, with_loss))
        
        y = self._get_y(average)

        self._sketch.clear_fill()
        self._sketch.set_stroke(const.INACTIVE_TEXT_COLOR)
        self._sketch.set_stroke_weight(1)
        self._sketch.draw_line(80, y, self._width - 5, y)

        self._sketch.clear_stroke()
        self._sketch.set_fill(const.INACTIVE_TEXT_COLOR)
        self._sketch.set_text_font(const.FONT_SRC, 12)
        self._sketch.set_text_align('right', 'bottom')
        self._sketch.draw_text(self._width - 5, y - 3, 'Avg yield: %d' % round(average))

        if losses > 0:
            self._sketch.set_fill(const.HISTORY_BODY_LOSS_COLOR)
            self._sketch.set_text_align('right', 'top')
            self._sketch.draw_text(self._width - 5, y + 3, 'Loss events: %d' % losses)

        self._sketch.pop_style()
        self._sketch.pop_transform()


    def _get_x(self, year):
        return 80 + (self._width - 80 - 100) * (year - 1) / 10

    def _get_y(self, value):
        offset = 40 + (self._height - 50) * value / 400
        return self._height - offset

    def _invert_y(self, y):
         return ((self._height - y - 40) * 400) / (self._height - 50)


def main():
    presenter = HistoryMainPresenter('History Viz', None)


if __name__ == '__main__':
    main()
