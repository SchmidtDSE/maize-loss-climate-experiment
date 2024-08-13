"""Supporting utilities for common UI elements.

License:
    BSD
"""
import math

import const


class ToggleButtonSet:
    """A set of buttons representing choices where only one can be selected at a time."""

    def __init__(self, sketch, x, y, label, options, selected, on_change, make_rows=True,
        narrow=False, show_label=False, keyboard_button=None):
        """Create a new toggle button set.

        Args:
            sketch: The Sketchingpy sketch in which this widget will operate.
            x: The x coordinate at which the widget should be drawn.
            y: The y coordinate at which the widget should be drawn.
            label: The human readable label describing the question or option for which the toggle
                buttons are provided.
            options: List of string options the user can select for the configuration.
            selected: The initially selected configuration option. Should appear in the options
                string list.
            on_change: Function to invoke when the option slected by the user changes, taking in a
                single string argument which is the new option selected.
            make_rows: Flag indicating if the buttons should appear in a grid where each row has
                three options. True will make the grid and false will leave the buttons
                horizontally adjacent regardless of the option count. Defaults to true.
            narrow: Flag indicating if the button widths should be made smaller. True if this
                compact configuration should be used and false otherwise. Defaults to false.
            show_label: Flag indicating if the label should be shown to the user. True if it should
                be shown and false otherwise.
            keyboard_button: Keyboard button which, when pressed should cycle between toggle button
                options or None if no keyboard button should cycle between choices.
        """
        self._sketch = sketch
        self._x = x
        self._y = y
        self._label = label
        self._options = options
        self._on_change = on_change
        self._selected = selected
        self._make_rows = make_rows
        self._narrow = narrow
        self._show_label = show_label
        self._keyboard_button = keyboard_button

    def set_value(self, option):
        """Update the value selected by the user.

        Update the value selected by the user, changing internal state and the display of this
        widget but not causing the callback function to be invoked.

        Args:
            option: String option to use as the new selected value.
        """
        self._selected = option

    def step(self, mouse_x, mouse_y, clicked, keypress=None):
        """Update and redraw this button.

        Args:
            mouse_x: The horizontal coordinate of the cursor.
            mouse_y: The vertical coordinate of the cursor.
            clicked: Flag indicating if the the mouse button has been pressed since step was last
                called. True if the button was pressed and false otherwise.
            keypress: The key pressed since step was last called or None if no key pressed. A
                string if the button was pressed and None otherwise.
        """
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.translate(self._x, self._y)

        self._sketch.set_stroke_weight(1)
        self._sketch.set_rect_mode('corner')
        self._sketch.set_text_font(const.FONT_SRC, 14)

        button_x = 0
        button_y = 20 if self._show_label else 0
        mouse_x = mouse_x - self._x
        mouse_y = mouse_y - self._y

        if self._show_label:
            self._sketch.clear_stroke()
            self._sketch.set_fill(const.INACTIVE_TEXT_COLOR)
            self._sketch.set_text_align('left', 'baseline')
            self._sketch.draw_text(0, 14, self._label)

        self._sketch.set_text_align('center', 'baseline')

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

        many_options = len(self._options) > 2

        if self._narrow:
            button_width = const.BUTTON_WIDTH_NARROW
        else:
            button_width = const.BUTTON_WIDTH_COMPACT if many_options else const.BUTTON_WIDTH

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

        if self._keyboard_button is not None and self._keyboard_button == keypress:
            index = self._options.index(self._selected)

            new_index = index + 1
            if new_index >= len(self._options):
                new_index = 0

            option = self._options[new_index]
            self._selected = option
            self._on_change(option)

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def get_height(self):
        """Get how much vertical space this widget takes up.

        Returns:
            Height of this widget in pixels.
        """
        if self._make_rows:
            rows = math.ceil(len(self._options) / 3)
            height = const.BUTTON_HEIGHT * rows
        else:
            height = const.BUTTON_HEIGHT

        if self._show_label:
            height += 20

        return height


class Button:
    """Widget which shows text and can be clicked on."""

    def __init__(self, sketch, x, y, label, on_click, narrow=False, keyboard_button=None):
        """Create a new button.

        Args:
            sketch: The sketchingpy sketch in which to build this widget.
            x: The horizontal coordinate at which the button should be made.
            y: The vertical coordinate at which the button should be made.
            label: The text to display on the button.
            on_click: Function to call with single argument (the label of this button) when this
                button is clicked. This will also be fired if the button is tapped.
            narrow: Flag indicating if the width of this button should be shortened. True if it
                should use the compact configuration and false if normal width.
            keyboard_button: The string button name that, when pressed, causes this button to act
                as if it was clicked.
        """
        self._sketch = sketch
        self._x = x
        self._y = y
        self._label = label
        self._on_click = on_click
        self._narrow = narrow
        self._keyboard_button = keyboard_button

    def step(self, mouse_x, mouse_y, clicked, keypress=None):
        """Update and draw this button.

        Args:
            mouse_x: The horizontal coordinate of the cursor.
            mouse_y: The vertical coordinate of the cursor.
            clicked: Flag indicating if the the mouse button has been pressed since step was last
                called. True if the button was pressed and false otherwise.
            keypress: The key pressed since step was last called or None if no key pressed. A
                string if the button was pressed and None otherwise.
        """
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

        button_pressed = (self._keyboard_button is not None) and self._keyboard_button == keypress

        if (is_hovering and clicked) or button_pressed:
            self._on_click(self._label)

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def get_height(self):
        """Get how much vertical space this widget takes up.

        Returns:
            Height of this widget in pixels.
        """
        return const.BUTTON_HEIGHT
