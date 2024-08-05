import math

import const


class ToggleButtonSet:

    def __init__(self, sketch, x, y, label, options, selected, on_change, make_rows=True,
        narrow=False, show_label=False, keyboard_button=None):
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
        self._selected = option

    def step(self, mouse_x, mouse_y, clicked, keypress=None):
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
        if self._make_rows:
            rows = math.ceil(len(self._options) / 3)
            height = const.BUTTON_HEIGHT * rows
        else:
            height = const.BUTTON_HEIGHT

        if self._show_label:
            height += 20

        return height


class Button:

    def __init__(self, sketch, x, y, label, on_click, narrow=False, keyboard_button=None):
        self._sketch = sketch
        self._x = x
        self._y = y
        self._label = label
        self._on_click = on_click
        self._narrow = narrow
        self._keyboard_button = keyboard_button

    def step(self, mouse_x, mouse_y, clicked, keypress=None):
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
        return const.BUTTON_HEIGHT
