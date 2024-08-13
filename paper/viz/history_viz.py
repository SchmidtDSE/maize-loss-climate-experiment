"""Visualization of a single APH and claims under different regulatory schemes.

Visualization of a single APH and claims under different regulatory schemes with a small window
into system-wide effects of adopting that scheme. This is the "claims visualization" as named in
the paper.

License:
    BSD
"""

import copy
import statistics
import sys

import sketchingpy

import buttons
import const

HIGH_VARIABILITY_SCENARIO = [230, 150, 260, 210, 290, 155, 290, 190, 310, 35]
LOW_VARIABILITY_SCENARIO = [190, 200, 190, 195, 200, 210, 200, 205, 220, 160]

NUM_ARGS = 4
USAGE_STR = 'python history_viz.py [csv location] [output location] [threshold] [scenario]'


class SummaryDataFacade:
    """Facade allowing for access to data summarizing the impact of a policy system-wide."""

    def __init__(self, raw_rows):
        """Create a new data model facade.
        
        Args:
            raw_rows: The rows around which this facade will simplify access. These will be parsed
                with expected data types.
        """
        with_offset = filter(lambda x: x['offsetBaseline'] == 'always', raw_rows)
        with_size = filter(lambda x: x['geohashSimSize'] == '4.0', with_offset)
        with_threshold = filter(lambda x: x['threshold'] == '0.25', with_size)
        parsed = map(lambda x: self._parse_record(x), with_threshold)
        parsed_tuple = map(
            lambda x: (self._get_key(x['condition'], x['year']), x),
            parsed
        )
        self._inner_records = dict(parsed_tuple)

    def get_claims(self, year, condition, using_std):
        """Get the claims rate for a year, simulation condition, and regulatory scheme.
        
        Args:
            year: Get the year for which the claims rate is requested.
            condition: The condition like historic or 2050_SSP245.
            using_std: True if using a standard deviation-based threshold or false if using the
                average-based threshold.
        
        Returns:
            The claims rate (0 - 100) for the simulation configuration requested.
        """
        key = self._get_key(condition, year)
        attr = 'claimsRate' + ('Std' if using_std else '')
        return self._inner_records[key][attr]

    def get_threshold(self, year, condition, using_std):
        """Get the loss threshold associated with a threshold.
        
        Args:
            year: The year in which the the data are requested.
            condition: The condition like historic or 2050_SSP245.
            using_std: True if using a standard deviation-based threshold or false if using the
                average-based threshold.
        
        Returns:
            The loss threshold used for the simulation requested.
        """
        key = self._get_key(condition, year)
        attr = 'threshold' + ('Std' if using_std else '')
        return self._inner_records[key][attr]

    def _parse_record(self, target):
        """Parse a raw record.
        
        Args:
            target: The raw dictionary to parse.
        
        Returns:
            Parsed record.
        """
        return {
            'year': int(target['year']),
            'condition': target['condition'],
            'threshold': float(target['threshold']),
            'thresholdStd': float(target['thresholdStd']),
            'claimsRate': float(target['claimsRate']),
            'claimsRateStd': float(target['claimsRateStd'])
        }

    def _get_key(self, condition, year):
        """Get a key refering to a combination of simulated condition and year.
        
        Args:
            condition: The condition like future or 2050_SSP245 for which a key is being generated.
            year: The year for which a key is being generated.
        
        Returns:
            Key describing a year within a condition.
        """
        return '%s_%d' % (condition, year)


class HistoryMainPresenter:
    """Presenter running the history viz."""

    def __init__(self, target, loading_id, csv_loc=None, output_loc=None,
        default_threshold='average', default_scenario='high'):
        """Create a new presenter.
        
        Args:
            target: The title of of the window for the visualization or the HTML ID in which the
                visualization should be drawn.
            loading_id: The ID of the HTML element where the loading indicator can be found which
                should be hidden after the visualization is loaded. Ignored if not running in web.
            csv_loc: The path to the CSV file with the system-wide data to be used to contextualize
                chosen regulatory structures. If None, will use a default. Defaults to None.
            output_loc: The location at which the visualization should be written or None if the
                visualization should run interactively. Defaults to None.
            default_threshold: The regulatory scheme to use like 'average' for average-based
                thresholds. Defaults to average.
            default_scenario: The scenario to be shown like 2050_SSP245 when offering contextual
                information.
        """
        if output_loc:
            self._sketch = sketchingpy.Sketch2DStatic(const.WIDTH, const.HEIGHT)
        else:
            self._sketch = sketchingpy.Sketch2D(const.WIDTH, const.HEIGHT, target, loading_id)

        self._sketch.set_fps(10)

        self._click_waiting = False
        self._key_waiting = None

        if csv_loc is None:
            csv_loc = './data/export_claims.csv'

        raw_rows = self._sketch.get_data_layer().get_csv(csv_loc)
        self._data_facade = SummaryDataFacade(raw_rows)

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

        self._summary_presenter = SummaryPresenter(
            self._sketch,
            const.WIDTH - 219,
            30,
            200,
            100,
            self._data_facade
        )

        threshold_label = 'Average-based' if default_threshold == 'average' else 'Stdev-based'
        self._threshold_type_buttons = buttons.ToggleButtonSet(
            self._sketch,
            5,
            const.HEIGHT - (const.BUTTON_HEIGHT + 7),
            'Threshold',
            ['Average-based', 'Stdev-based'],
            threshold_label,
            lambda new_val: self._update_threshold(new_val),
            keyboard_button='t'
        )
        self._update_threshold(threshold_label)

        scenario_label = 'High Stability' if default_scenario == 'high' else 'Low Stability'
        self._scenario_buttons = buttons.ToggleButtonSet(
            self._sketch,
            const.WIDTH - 10 - const.BUTTON_WIDTH * 2,
            const.HEIGHT - (const.BUTTON_HEIGHT + 7),
            'Example',
            ['Low Stability', 'High Stability'],
            scenario_label,
            lambda new_val: self._update_stability(new_val),
            keyboard_button='s'
        )
        self._update_stability(scenario_label)

        if output_loc:
            self._step()
            self._sketch.save_image(output_loc)
        else:
            mouse = self._sketch.get_mouse()

            def set_mouse_clicked(mouse):
                self._click_waiting = True

            mouse.on_button_press(set_mouse_clicked)

            keyboard = self._sketch.get_keyboard()

            def set_key_waiting(button):
                self._key_waiting = button.get_name()

            keyboard.on_key_press(set_key_waiting)

            self._sketch.on_step(lambda x: self._step())
            self._sketch.show()

    def _step(self):
        """Check for changes and optionally redraw the visualization."""
        mouse = self._sketch.get_mouse()

        if mouse:
            mouse_x = mouse.get_pointer_x()
            mouse_y = mouse.get_pointer_y()
        else:
            mouse_x = 0
            mouse_y = 0

        mouse_x_same = mouse_x == self._last_mouse_x
        mouse_y_same = mouse_y == self._last_mouse_y
        click_clear = (not self._click_waiting) and (self._key_waiting is None)
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
        self._summary_presenter.step(mouse_x, mouse_y, self._click_waiting)
        self._threshold_type_buttons.step(mouse_x, mouse_y, self._click_waiting, self._key_waiting)
        self._scenario_buttons.step(mouse_x, mouse_y, self._click_waiting, self._key_waiting)

        self._sketch.pop_style()
        self._sketch.pop_transform()

        self._click_waiting = False
        self._key_waiting = None

    def _draw_annotation(self):
        """Draw a label with instructions."""
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.set_fill('#333333')
        self._sketch.clear_stroke()
        self._sketch.set_text_font(const.FONT_SRC, 12)
        self._sketch.set_text_align('left', 'bottom')
        self._sketch.draw_text(5, 17, const.HISTORY_INSTRUCTION)

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _update_threshold(self, new_value, update_buttons=False):
        """Update the loss threshold type.
        
        Args:
            new_value: Description of the regulatory structure where 'Stdev-based' uses the
                standard deviation-based thresholds.
            update_buttons: Flag indicating if the buttons for this parameter should be updated.
                Defaults to false.
        """
        self._change_waiting = True
        using_std = new_value == 'Stdev-based'
        self._chart_presenter.set_using_std(using_std)
        self._summary_presenter.set_using_std(using_std)
        
        if update_buttons:
            self._threshold_type_buttons.set_value(new_value)

    def _update_stability(self, new_setting):
        """Update the stability setting.
        
        Args:
            new_value: Description of the stability level like 'Low Stability' to use as an APH
                starting point that the user can further modify.
            update_buttons: Flag indicating if the buttons for this parameter should be updated.
                Defaults to false.
        """
        self._change_waiting = True
        if new_setting == 'Low Stability':
            self._chart_presenter.set_values(HIGH_VARIABILITY_SCENARIO)
        else:
            self._chart_presenter.set_values(LOW_VARIABILITY_SCENARIO)
        
        if update_buttons:
            self._scenario_buttons.set_value(new_setting)


class HistoryChartPresenter:
    """Presenter running the modifiable APH chart."""

    def __init__(self, sketch, x, y, width, height, start_values):
        """Create a new APH presenter.
        
        Args:
            sketch: The Sketchingpy sketch in which the chart should be created.
            x: The horizontal position at which the chart should be created.
            y: The vertical position at which the chart should be created.
            width: The horizontal size in pixels of the chart.
            height: The vertical size in pixels of the chart.
            start_values: The starting APH values to show in the chart.
        """
        self._sketch = sketch
        self._use_std = False
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._values = start_values

    def step(self, mouse_x_abs, mouse_y_abs, clicked):
        """Update and redraw this chart.
        
        Args:
            mouse_x_abs: The x coordinate of the cursor.
            mouse_y_abs: The y coordinate of the cursor.
            clicked: True if a mouse button has been pressed since the last call to step or false
                if no mouse button press since the last invocation.
        """
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

        self._sketch.set_text_font(const.FONT_SRC, 16)
        self._sketch.set_text_align('center', 'center')
        self._sketch.clear_stroke()
        self._sketch.set_fill(const.ACTIVE_TEXT_COLOR)
        self._sketch.draw_text(self._width / 2, 14, 'Simulated Individual Farm Production History')

        self._draw_hover(mouse_x, mouse_y)
        self._draw_candidate(mouse_x, mouse_y, clicked)
        self._draw_x_axis()
        self._draw_y_axis()
        self._draw_average()
        self._draw_content()

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def set_using_std(self, new_use_std):
        """Update if a standard deviation-based threshold should be used.
        
        Args:
            new_use_std: True if a standard deviation-based threshold should be used or false if
                an average-based threshold should be used.
        """
        self._use_std = new_use_std

    def set_values(self, new_values):
        """Set the APH yield values.
        
        Args:
            new_values: Yield values per year.
        """
        self._values = new_values

    def _get_year(self, mouse_x, mouse_y):
        """Get the year corresponding to a mouse position.
        
        Args:
            mouse_x: The horizontal position of the cursor.
            mouse_y: The vertical position of the cursor.
        
        Returns:
            Year corresponding to the cursor position.
        """
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
        """Draw a hover indicator below the cursor.
        
        Args:
            mouse_x: The horizontal position of the cursor.
            mouse_y: The vertical position of the cursor.
        """
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
        """Draw the horizontal axis showing years."""
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
        """Draw the vertical axis showing yield."""
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
        """Draw a potential APH change below the cursor.
        
        Args:
            mouse_x_abs: The x coordinate of the cursor.
            mouse_y_abs: The y coordinate of the cursor.
            clicked: True if a mouse button has been pressed since the last call to step or false
                if no mouse button press since the last invocation.
        """
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
        """Draw the content of the chart."""
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
        """Draw indicator of average of yields displayed."""
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
        """Get the x position corresponding to a year.
        
        Args:
            year: The year for which to find an x position.
        
        Returns:
            Horizontal coordinate of a year.
        """
        return 80 + (self._width - 80 - 100) * (year - 1) / 10

    def _get_y(self, value):
        """Get the y position corresponding to a yield.
        
        Args:
            value: The yield level.
        
        Returns:
            Vertical coordinate of a yield.
        """
        offset = 40 + (self._height - 50) * value / 400
        return self._height - offset

    def _invert_y(self, y):
        """Convert from a y position to a yield level.
        
        Args:
            y: The vertical coordinate.
        
        Returns:
            Yield level corresponding to the coordinate.
        """
        return ((self._height - y - 40) * 400) / (self._height - 50)


class SummaryPresenter:
    """Small visualization element showing overall impact of a regulatory scheme.
    
    Small visualization element showing overall impact of a regulatory scheme if it were applied
    system-wide using historic data.
    """

    def __init__(self, sketch, x, y, width, height, data_facade):
        """Create new presenter.
        
        Args:
            sketch: The Sketchingpy sketch in which the summary should be created.
            x: The horizontal position at which the summary should be made.
            y: The vertical position at which the summary should be made.
            width: The horizontal size of the element.
            height: The vertical size of the element.
            data_facade: Facade through which system-wide data can be accessed.
        """
        self._sketch = sketch
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._data_facade = data_facade
        self._using_std = False

    def set_using_std(self, using_std):
        """Change if the standard deviation-based threshold should be used.
        
        Args:
            using_std: True if a standard deviation-based loss threshold should be used or false if
                average-based loss threshold should be used.
        """
        self._using_std = using_std

    def step(self, mouse_x_abs, mouse_y_abs, clicked):
        """Update and redraw this component.
        
        Args:
            mouse_x_abs: The horizontal position of the cursor.
            mouse_y_abs: The vertical position of the cursor.
            clicked: True if the mouse button has been pressed since the last call to step or false
                otherwise.
        """
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.translate(self._x, self._y)

        # Draw axis and title
        self._sketch.set_fill(const.PANEL_BG_COLOR + 'A0')
        self._sketch.set_stroke(const.INACTIVE_BORDER)
        self._sketch.set_stroke_weight(1)
        self._sketch.set_rect_mode('corner')
        self._sketch.draw_rect(0, 0, self._width, self._height)

        self._sketch.clear_stroke()
        self._sketch.set_fill(const.ACTIVE_TEXT_COLOR)
        self._sketch.set_text_font(const.FONT_SRC, 12)
        self._sketch.set_text_align('center', 'center')
        self._sketch.draw_text(self._width / 2, 10, 'US Corn Belt Summary')

        self._sketch.clear_fill()
        self._sketch.set_stroke(const.INACTIVE_BORDER)
        self._sketch.draw_line(4, self._height - 18, self._width - 4, self._height - 18)

        self._sketch.clear_stroke()
        self._sketch.set_fill(const.INACTIVE_TEXT_COLOR)
        self._sketch.set_text_align('center', 'top')
        self._sketch.draw_text(self._width / 2, self._height - 16, 'Claims Rate')

        self._sketch.set_text_align('left', 'top')
        self._sketch.draw_text(4, self._height - 16, '0%')

        self._sketch.set_text_align('right', 'top')
        self._sketch.draw_text(self._width - 4, self._height - 16, '10%')

        # Draw top bar
        historic_claims = self._data_facade.get_claims(2010, 'historic', self._using_std) * 100
        historic_threshold = self._data_facade.get_threshold(2010, 'historic', self._using_std)
        future_claims = self._data_facade.get_claims(2050, '2050_SSP245', self._using_std) * 100
        future_threshold = self._data_facade.get_threshold(2050, '2050_SSP245', self._using_std)
        threshold_descriptor = ' std' if self._using_std else '% of avg'

        self._sketch.clear_stroke()
        self._sketch.set_fill(const.INACTIVE_TEXT_COLOR)
        self._sketch.set_text_font(const.FONT_SRC, 12)
        self._sketch.set_text_align('left', 'bottom')

        historic_str_vals = (historic_claims, historic_threshold, threshold_descriptor)
        self._sketch.draw_text(2, 40, 'Historic: %.1f%% (<-%.2f%s)' % historic_str_vals)

        future_str_vals = (future_claims, future_threshold, threshold_descriptor)
        self._sketch.draw_text(2, 64, 'Future: %.1f%% (<-%.2f%s)' % future_str_vals)

        self._sketch.set_rect_mode('corner')
        get_width = lambda x: x / 10 * (self._width - 8)
        self._sketch.draw_rect(4, 41, get_width(historic_claims), 3)
        self._sketch.draw_rect(4, 65, get_width(future_claims), 3)

        self._sketch.pop_style()
        self._sketch.pop_transform()


def main():
    """Entrypoint into this visualization.
    
    Entrypoint into this visualization which runs interactively if no arguments provided or draws
    into a file if arguments provided.
    """
    
    if len(sys.argv) == 1:
        presenter = HistoryMainPresenter('History Viz', None)
    elif len(sys.argv) != NUM_ARGS + 1:
        print(USAGE_STR)
        sys.exit(1)
    else:
        csv_loc = sys.argv[1]
        output_loc = sys.argv[2]
        threshold = sys.argv[3]
        scenario = sys.argv[4]
        presenter = HistoryMainPresenter(
            'History Viz',
            None,
            csv_loc=csv_loc,
            output_loc=output_loc,
            default_threshold=threshold,
            default_scenario=scenario
        )

    assert presenter is not None


if __name__ == '__main__':
    main()
