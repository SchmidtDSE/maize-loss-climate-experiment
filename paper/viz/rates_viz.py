"""Visualization of hypothetical rate setting (rates visualization).

License:
    BSD
"""
import sketchingpy

import buttons
import const


class RatesConfig:
    """Configuration object describing what rates scenario the user has established."""

    def __init__(self, aph, county_yield, price, county_rate, subsidy, coverage, aph_type,
        perspective):
        """Create a new rates configuration.

        Args:
            aph: The average yield (APH).
            county_yield: Reference county-level average yield.
            price: The expected spring price for the crop.
            county_rate: The cost to insure one dollar of crop in dollars.
            subsidy: The percent of the cost subsidized by the government (0 - 1).
            aph_type: The type of APH threshold to use like 'Average-based' which uses a
                traditional percent below historic average loss threshold.
            coverage: The coverage level like 0.75 for liss threshold of 25%.
            perspective: Showing perspective from grower or grovernment where 'Subsidy' uses the
                government perspective.
        """
        self._aph = aph
        self._county_yield = county_yield
        self._price = price
        self._county_rate = county_rate
        self._subsidy = subsidy
        self._coverage = coverage
        self._aph_type = aph_type
        self._perspective = perspective

    def get_aph(self):
        """Get the average expected yield.

        Returns:
            The average yield (APH).
        """
        return self._aph

    def get_with_aph(self, aph):
        """Make a copy of this configuration with a new average yield.

        Args:
            aph: The new value to use.

        Returns:
            A copy of this record with the new value.
        """
        return RatesConfig(
            aph,
            self.get_county_yield(),
            self.get_price(),
            self.get_county_rate(),
            self.get_subsidy(),
            self.get_coverage(),
            self.get_aph_type(),
            self.get_perspective()
        )

    def get_county_yield(self):
        """Get the county average yield.

        Returns:
            Reference county-level average yield.
        """
        return self._county_yield

    def get_with_county_yield(self, county_yield):
        """Make a copy of this record with a new county average yield.

        Args:
            county_yield: The new value to use.

        Returns:
            A copy of this record with the new value.
        """
        return RatesConfig(
            self.get_aph(),
            county_yield,
            self.get_price(),
            self.get_county_rate(),
            self.get_subsidy(),
            self.get_coverage(),
            self.get_aph_type(),
            self.get_perspective()
        )

    def get_price(self):
        """Get the expected sale price for the crop.

        Returns:
            The expected spring price for the crop.
        """
        return self._price

    def get_with_price(self, price):
        """Make a copy of this record with a new expected sale price.

        Args:
            price: The new value to use.

        Returns:
            A copy of this record with the new value.
        """
        return RatesConfig(
            self.get_aph(),
            self.get_county_yield(),
            price,
            self.get_county_rate(),
            self.get_subsidy(),
            self.get_coverage(),
            self.get_aph_type(),
            self.get_perspective()
        )

    def get_county_rate(self):
        """Get the county insurance rate.

        Returns:
            The cost to insure one dollar of crop in dollars.
        """
        return self._county_rate

    def get_with_county_rate(self, county_rate):
        """Make a copy of this object with a new county insurance rate.

        Args:
            county_rate: The new value to use.

        Returns:
            A copy of this record with the new value.
        """
        return RatesConfig(
            self.get_aph(),
            self.get_county_yield(),
            self.get_price(),
            county_rate,
            self.get_subsidy(),
            self.get_coverage(),
            self.get_aph_type(),
            self.get_perspective()
        )

    def get_subsidy(self):
        """Get the percent subsidized by the government.

        Returns:
            The percent of the cost subsidized by the government (0 - 1).
        """
        return self._subsidy

    def get_with_subsidy(self, subsidy):
        """Make a copy of this record with a new subsidy level.

        Args:
            subsidy: The new value to use.

        Returns:
            A copy of this record with the new value.
        """
        return RatesConfig(
            self.get_aph(),
            self.get_county_yield(),
            self.get_price(),
            self.get_county_rate(),
            subsidy,
            self.get_coverage(),
            self.get_aph_type(),
            self.get_perspective()
        )

    def get_coverage(self):
        """Get the coverage level of the policy simulated.

        Returns:
            The coverage level like 0.75 for liss threshold of 25%.
        """
        return self._coverage

    def get_with_coverage(self, coverage):
        """Make a copy of this record with a new coverage level.

        Args:
            coverage: The new value to use.

        Returns:
            A copy of this record with the new value.
        """
        return RatesConfig(
            self.get_aph(),
            self.get_county_yield(),
            self.get_price(),
            self.get_county_rate(),
            self.get_subsidy(),
            coverage,
            self.get_aph_type(),
            self.get_perspective()
        )

    def get_aph_type(self):
        """Get the type of APH threshold used.

        Returns:
            The type of APH threshold to use like 'Average-based' which uses a traditional percent
            below historic average loss threshold.
        """
        return self._aph_type

    def get_with_aph_type(self, type):
        """Make a new copy of this configuration with a new APH type.

        Args:
            type: The new value to use.

        Returns:
            A copy of this record with the new value.
        """
        return RatesConfig(
            self.get_aph(),
            self.get_county_yield(),
            self.get_price(),
            self.get_county_rate(),
            self.get_subsidy(),
            self.get_coverage(),
            type,
            self.get_perspective()
        )

    def get_perspective(self):
        """Get the perspective from which prices are currently shown.

        Returns:
            Showing perspective from grower or grovernment where 'Subsidy' uses the government
            perspective.
        """
        return self._perspective

    def get_with_perspective(self, perspective):
        """Make a copy of this object with a new perspective setting.

        Args:
            perspective: The new value to use.

        Returns:
            A copy of this record with the new value.
        """
        return RatesConfig(
            self.get_aph(),
            self.get_county_yield(),
            self.get_price(),
            self.get_county_rate(),
            self.get_subsidy(),
            self.get_coverage(),
            self.get_aph_type(),
            perspective
        )


class RatesMainPresenter:
    """Presenter to run the rates visualization."""

    def __init__(self, target, loading_id):
        """Create a new rates visualization.

        Args:
            target: The ID in which the visualization should be initalized or the title of the
                window if not running on web.
            loading_id: The ID of the loading indicator to hide after initizalization.
        """
        self._sketch = sketchingpy.Sketch2D(const.WIDTH, const.HEIGHT, target, loading_id)

        self._sketch.set_fps(10)

        self._click_waiting = False
        self._key_waiting = None

        self._last_mouse_x = None
        self._last_mouse_y = None
        self._change_waiting = False

        start_config = RatesConfig(
            200,
            175,
            4,
            0.04,
            0.55,
            0.75,
            'Average-based',
            'Subsidy'
        )

        self._chart_presenter = RatesChartPresenter(
            self._sketch,
            5,
            20,
            const.WIDTH - 320,
            const.HEIGHT - 5 - 20,
            start_config
        )

        self._config_presenter = ConfigPresenter(
            self._sketch,
            const.WIDTH - 305,
            5,
            lambda config: self._update_config(config),
            start_config
        )

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
        """Update the visualization and redraw if needed."""
        mouse = self._sketch.get_mouse()

        if mouse:
            mouse_x = mouse.get_pointer_x()
            mouse_y = mouse.get_pointer_y()
        else:
            mouse_x = 0
            mouse_y = 0

        mouse_x_same = mouse_x == self._last_mouse_x
        mouse_y_same = mouse_y == self._last_mouse_y
        click_clear = (not self._click_waiting) and (not self._key_waiting)
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
        self._config_presenter.step(mouse_x, mouse_y, self._click_waiting, self._key_waiting)

        self._sketch.pop_style()
        self._sketch.pop_transform()

        self._click_waiting = False
        self._key_waiting = None

    def _draw_annotation(self):
        """Draw instructional text for the user."""
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.set_fill('#333333')
        self._sketch.clear_stroke()
        self._sketch.set_text_font(const.FONT_SRC, 12)
        self._sketch.set_text_align('left', 'bottom')
        self._sketch.draw_text(5, 17, const.RATES_INSTRUCTION)

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _update_config(self, config):
        """Update the configuration of the hypothetical rate setting."""
        self._chart_presenter.update_config(config)
        self._change_waiting = True


class RatesChartPresenter:
    """Chart showing the rate setting selected by the user."""

    def __init__(self, sketch, x, y, width, height, start_config):
        """Create a new rates chart.

        Args:
            sketch: The Sketchingpy sketch in which to create the rates chart.
            x: The horizontal coordinate at which to build the chart.
            y: The vertical coordinate at which to build the chart.
            width: The horizontal size of the rates chart in pixels.
            height: The vertical size of the rates chart in pixels.
            start_config: Starting RatesConfig.
        """
        self._sketch = sketch
        self._use_std = False
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._config = start_config

    def step(self, mouse_x_abs, mouse_y_abs, clicked):
        """Update and redraw this component.

        Args:
            mouse_x_abs: The horizontal position of the cursor.
            mouse_y_abs: The vertical position of the cursor.
            clicked: True if the mouse button was clicked since the last call to step or false
                otherwise.
        """
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.translate(self._x, self._y)

        self._sketch.set_fill(const.PANEL_BG_COLOR)
        self._sketch.set_stroke(const.INACTIVE_BORDER)
        self._sketch.set_stroke_weight(1)
        self._sketch.set_rect_mode('corner')
        self._sketch.draw_rect(0, 0, self._width, self._height)

        self._sketch.set_text_font(const.FONT_SRC, 16)
        self._sketch.set_text_align('center', 'center')
        self._sketch.clear_stroke()
        self._sketch.set_fill(const.ACTIVE_TEXT_COLOR)
        self._sketch.draw_text(self._width / 2, 18, 'Hypothetical Rates Simulator')

        self._draw_x_axis()
        self._draw_y_axis()
        self._draw_content()

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def update_config(self, config):
        """Update the rates setting configuration.

        Args:
            config: New RatesConfig.
        """
        self._config = config

    def _draw_x_axis(self):
        """Draw the horizontal axis showing loss threshold."""
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.set_text_align('center', 'top')
        self._sketch.set_fill(const.INACTIVE_TEXT_COLOR)
        self._sketch.clear_stroke()
        self._sketch.set_text_font(const.FONT_SRC, 12)

        if self._config.get_aph_type() == 'Average-based':
            self._sketch.draw_text(
                self._get_x(30),
                self._height - 14,
                'Loss Threshold (% Below APH)'
            )

            for percent in range(15, 55, 5):
                percent_str = '-%d%%' % percent
                self._sketch.draw_text(self._get_x(percent), self._height - 28, percent_str)
        else:
            self._sketch.draw_text(
                self._get_x(30),
                self._height - 14,
                'Loss Threshold (Std Below APH)'
            )

            for percent in range(15, 55, 5):
                percent_str = '-%.2f std' % (percent * 1.5 / 25)
                self._sketch.draw_text(self._get_x(percent), self._height - 28, percent_str)

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _draw_y_axis(self):
        """Draw the vertical axis showing price."""
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.set_text_align('right', 'center')
        self._sketch.set_fill(const.INACTIVE_TEXT_COLOR)
        self._sketch.clear_stroke()
        self._sketch.set_text_font(const.FONT_SRC, 12)

        for value in range(0, 35, 5):
            self._sketch.draw_text(35, self._get_y(value), value)

        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.translate(13, self._get_y(15))
        self._sketch.set_angle_mode('degrees')
        self._sketch.rotate(-90)
        self._sketch.set_text_align('center', 'center')

        if self._config.get_perspective() == 'Subsidy':
            self._sketch.draw_text(0, 0, 'Government Subsidy ($ / acre)')
        else:
            self._sketch.draw_text(0, 0, 'Grower Price ($ / acre)')

        self._sketch.pop_style()
        self._sketch.pop_transform()

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _draw_content(self):
        """Draw the content of the chart."""
        self._sketch.push_transform()
        self._sketch.push_style()

        grower_y_start = self._get_y(self._get_price(True, 0.5))
        grower_y_end = self._get_y(self._get_price(True, 0.85))

        county_y_start = self._get_y(self._get_price(False, 0.50))
        county_y_end = self._get_y(self._get_price(False, 0.85))

        self._sketch.set_text_font(const.FONT_SRC, 12)
        self._sketch.set_text_align('left', 'center')
        self._sketch.clear_stroke()
        self._sketch.set_fill('#A0A0A0')
        self._sketch.draw_text(
            self._get_x(50),
            county_y_start + (10 if grower_y_start < county_y_start else -15),
            'County'
        )

        self._sketch.clear_fill()
        self._sketch.set_stroke_weight(3)
        self._sketch.set_stroke('#A0A0A0')
        self._sketch.draw_line(
            self._get_x(50),
            county_y_start,
            self._get_x(15),
            county_y_end
        )

        self._sketch.set_text_font(const.FONT_SRC, 12)
        self._sketch.set_text_align('right', 'center')
        self._sketch.clear_stroke()
        self._sketch.set_fill('#505050')
        self._sketch.draw_text(
            self._get_x(15),
            grower_y_end + (15 if grower_y_start < county_y_start else -10),
            'Grower'
        )

        self._sketch.clear_fill()
        self._sketch.set_stroke_weight(3)
        self._sketch.set_stroke('#505050')
        self._sketch.draw_line(
            self._get_x(50),
            grower_y_start,
            self._get_x(15),
            grower_y_end
        )

        policy_price = self._get_price(True, self._config.get_coverage())
        policy_coverage = self._config.get_coverage()
        policy_x = self._get_x((1 - policy_coverage) * 100)
        policy_y = self._get_y(policy_price)

        self._sketch.clear_stroke()
        self._sketch.set_fill('#505050')
        self._sketch.set_ellipse_mode('radius')
        self._sketch.draw_ellipse(policy_x, policy_y, 5, 5)

        self._sketch.clear_fill()
        self._sketch.set_stroke('#505050')
        self._sketch.set_stroke_weight(3)
        self._sketch.set_ellipse_mode('radius')
        self._sketch.draw_ellipse(policy_x, policy_y, 10, 10)

        self._sketch.set_stroke_weight(1)
        self._sketch.draw_line(policy_x, policy_y, policy_x, policy_y - 25)

        self._sketch.set_stroke('#505050')
        self._sketch.set_fill('#FFFFFFC0')
        self._sketch.set_rect_mode('corner')
        self._sketch.draw_rect(policy_x - 80, policy_y - 75, 160, 45)

        grower_price = self._get_price(True, self._config.get_coverage(), 'Grower Price')
        subsidy_price = self._get_price(True, self._config.get_coverage(), 'Subsidy')

        self._sketch.clear_stroke()
        self._sketch.set_fill('#505050')

        self._sketch.set_text_align('center', 'baseline')
        text_y = policy_y - 62
        self._sketch.draw_text(policy_x, text_y, 'After $%.2f / acre subsidy,' % subsidy_price)
        text_y += 14
        self._sketch.draw_text(
            policy_x,
            text_y,
            'insuring %.0f%% of APH' % (self._config.get_coverage() * 100)
        )
        text_y += 14
        self._sketch.draw_text(policy_x, text_y, 'costs grower $%.2f / acre.' % grower_price)

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _get_price(self, use_grower_aph, coverage, perspective=None):
        """Get the price associated with a hypothetical plan.

        Args:
            use_grower_aph: Flag indicting if the grower-level average should be used. True if the
                grower APH should set the price or false if the county average should be used.
            coverage: The coverage level for which the price should be set based on the APH type
                currently selected.
            perspective: String indicating if things are shown from the perspective of the grower
                or government. If not provided (defaults to None), will use the 'Subsidy' option
                instead of the 'Grower Price' option.
        """
        grower_aph = self._config.get_aph()
        county_aph = self._config.get_county_yield()
        aph = grower_aph if use_grower_aph else county_aph

        price = self._config.get_price()
        county_rate = self._config.get_county_rate()
        subsidy = self._config.get_subsidy()

        if perspective is None:
            perspective = self._config.get_perspective()

        if perspective == 'Grower Price':
            share = 1 - subsidy
        else:
            share = subsidy

        return aph * price * county_rate * coverage * share

    def _get_x(self, percent):
        """Get the horizontal position associated with a coverage level.

        Args:
            percent: The loss threshold converted to percent below average.

        Returns:
            Horizontal position in pixels.
        """
        chart_width = (self._width - 80 - 80)
        position_internal = chart_width * (percent - 15) / (50 - 15)
        position_internal_reverse = chart_width - position_internal
        return 80 + position_internal_reverse

    def _get_y(self, value):
        """Get the verticla position corresponding to a price.

        Args:
            value: The price to convert to a vertical coordinate.

        Returns:
            Vertical position in pixels.
        """
        offset = 40 + (self._height - 50) * value / 32
        return self._height - offset


class ConfigPresenter:
    """Presenter to modify the rates simulation configuration."""

    def __init__(self, sketch, x, y, on_config_change, start_config):
        """Create a new set of buttons within this meta-widget.

        Args:
            sketch: The Sketchingpy sketch in which to build this meta-widget.
            x: The horizontal position at which this meta-widget should be made.
            y: The vertical position at which this meta-widget should be made.
            on_config_change: Callback to invoke with a new RatesConfig when modified by the user.
            start_config: Initial RatesConfig.
        """
        self._sketch = sketch
        self._x = x
        self._y = y
        self._on_config_change = on_config_change

        y = 50
        self._aph_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            y,
            'Insured Unit APH (bushel / acre)',
            ['175', '200', '225'],
            '200',
            lambda x: self._change_aph(x),
            show_label=True,
            keyboard_button='i'
        )

        y += self._aph_buttons.get_height() + 30
        self._county_yield_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            y,
            'County average yield (bushel / acre)',
            ['150', '175', '200'],
            '175',
            lambda x: self._change_county_yield(x),
            show_label=True,
            keyboard_button='c'
        )

        y += self._county_yield_buttons.get_height() + 30
        self._price_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            y,
            'Projected price ($ / bushel)',
            [
                '$3',
                '$4',
                '$5'
            ],
            '$4',
            lambda x: self._change_price(x),
            show_label=True,
            keyboard_button='p'
        )

        y += self._price_buttons.get_height() + 30
        self._county_rate_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            y,
            'County premium rate (%)',
            [
                '3%',
                '4%',
                '5%'
            ],
            '4%',
            lambda x: self._change_county_rate(x),
            show_label=True,
            keyboard_button='r'
        )

        y += self._county_rate_buttons.get_height() + 30
        self._subsidy_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            y,
            'Subsidy rate (%)',
            [
                '55%',
                '60%',
                '65%'
            ],
            '55%',
            lambda x: self._change_subsidy(x),
            show_label=True,
            keyboard_button='s'
        )

        y += self._subsidy_buttons.get_height() + 30
        self._coverage_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            y,
            'Coverage amount (% of APH)',
            [
                '55%',
                '65%',
                '75%'
            ],
            '75%',
            lambda x: self._change_coverage(x),
            show_label=True,
            keyboard_button='a'
        )

        y += self._coverage_buttons.get_height() + 30
        self._type_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            y,
            'APH Type',
            [
                'Average-based',
                'Std-based'
            ],
            'Average-based',
            lambda x: self._change_type(x),
            show_label=True,
            keyboard_button='t'
        )

        y += self._type_buttons.get_height() + 30
        self._perspective_buttons = buttons.ToggleButtonSet(
            self._sketch,
            0,
            y,
            'Output',
            [
                'Subsidy',
                'Grower Price'
            ],
            'Subsidy',
            lambda x: self._change_perspective(x),
            show_label=True,
            keyboard_button='o'
        )

        self._config = start_config

    def step(self, mouse_x, mouse_y, click_waiting, keypress):
        """Update this visualization and redraw its components.

        Args:
            mouse_x: The horizontal position of the cursor.
            mouse_y: The vertical position of the cursor.
            click_waiting: True if the mouse button has been pressed since last calling step and
                false otherwise.
            keypress: String key pressed since last calling step and None if no key pressed since
                last call.
        """
        self._sketch.push_transform()
        self._sketch.push_style()

        mouse_x = mouse_x - self._x
        mouse_y = mouse_y - self._y

        self._sketch.translate(self._x, self._y)
        self._aph_buttons.step(mouse_x, mouse_y, click_waiting, keypress)
        self._county_yield_buttons.step(mouse_x, mouse_y, click_waiting, keypress)
        self._price_buttons.step(mouse_x, mouse_y, click_waiting, keypress)
        self._county_rate_buttons.step(mouse_x, mouse_y, click_waiting, keypress)
        self._subsidy_buttons.step(mouse_x, mouse_y, click_waiting, keypress)
        self._type_buttons.step(mouse_x, mouse_y, click_waiting, keypress)
        self._coverage_buttons.step(mouse_x, mouse_y, click_waiting, keypress)
        self._perspective_buttons.step(mouse_x, mouse_y, click_waiting, keypress)

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _change_aph(self, new_val):
        """Internal callback for when the average grower yield changed.

        Args:
            new_val: The new value selected by the user as a string matching the button.
        """
        val_interpreted = int(new_val)
        self._config = self._config.get_with_aph(val_interpreted)
        self._on_config_change(self._config)

    def _change_county_yield(self, new_val):
        """Internal callback for when the average county yield changed.

        Args:
            new_val: The new value selected by the user as a string matching the button.
        """
        val_interpreted = int(new_val)
        self._config = self._config.get_with_county_yield(val_interpreted)
        self._on_config_change(self._config)

    def _change_price(self, new_val):
        """Internal callback for when the expected price changed.

        Args:
            new_val: The new value selected by the user as a string matching the button.
        """
        val_interpreted = int(new_val.replace('$', ''))
        self._config = self._config.get_with_price(val_interpreted)
        self._on_config_change(self._config)

    def _change_county_rate(self, new_val):
        """Internal callback for when the county insurance rate changes.

        Args:
            new_val: The new value selected by the user as a string matching the button.
        """
        val_interpreted = float(new_val.replace('%', '')) / 100
        self._config = self._config.get_with_county_rate(val_interpreted)
        self._on_config_change(self._config)

    def _change_subsidy(self, new_val):
        """Internal callback for when the subsidy percentage changed.

        Args:
            new_val: The new value selected by the user as a string matching the button.
        """
        val_interpreted = float(new_val.replace('%', '')) / 100
        self._config = self._config.get_with_subsidy(val_interpreted)
        self._on_config_change(self._config)

    def _change_coverage(self, new_val):
        """Internal callback for when the coverage level changed.

        Args:
            new_val: The new value selected by the user as a string matching the button.
        """
        val_interpreted = float(new_val.replace('%', '')) / 100
        self._config = self._config.get_with_coverage(val_interpreted)
        self._on_config_change(self._config)

    def _change_type(self, new_val):
        """Internal callback for when the threshold type changed.

        Args:
            new_val: The new value selected by the user as a string matching the button.
        """
        val_interpreted = str(new_val)
        self._config = self._config.get_with_aph_type(val_interpreted)
        self._on_config_change(self._config)

    def _change_perspective(self, new_val):
        """Internal callback for when the price perspecrtive changed.

        Args:
            new_val: The new value selected by the user as a string matching the button.
        """
        val_interpreted = str(new_val)
        self._config = self._config.get_with_perspective(val_interpreted)
        self._on_config_change(self._config)


def main():
    """Main entry point for running this visualization interactively."""
    presenter = RatesMainPresenter('Rates Viz', None)
    assert presenter is not None


if __name__ == '__main__':
    main()
