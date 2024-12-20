"""Visualization which shows system-wide changes to yield and risk.

Visualization which shows system-wide changes to yield and risk under different scenarios and
simulation configurations. Called the "distributional visualization" in the paper.

License:
    BSD
"""

import functools
import sys

import sketchingpy

import buttons
import const

SERIES = ['predicted', 'counterfactual']
SUB_CHART_WIDTH = 700 - 10
SUB_CHART_HEIGHT = 200
TOP_COLOR = '#404040'
BOTTOM_COLOR = '#707070'
SUMMARY_FIELDS = ['claimsMpci', 'claimsSco', 'mean', 'cnt', 'claimsRate']

NUM_ARGS = 6
USAGE_PIECES = [
    '[csv location]',
    '[default year]',
    '[default coverage]',
    '[unit]',
    '[comparison]',
    '[output location]'
]
USAGE_STR = 'python hist_viz.py ' + (' '.join(USAGE_PIECES))


class MainPresenter:
    """Presenter at the root of the visualization driving all other components."""

    def __init__(self, target, loading_id=None, csv_loc=None, default_year=None,
        default_coverage=None, unit='unit risk', comparison='vs counterfact', output_loc=None):
        """Create a new main presenter.

        Args:
            target: The ID at which the visualization should be loaded or the window title if on
                desktop.
            loading_id: The ID with the loading indicator which should be hidden after
                initialization if on web. Ignored if not on web. Defaults to None.
            csv_loc: Location at which the source CSV summary of the "histogram" information
                displayed by this visualization should be found. Uses a default if None. Defaults
                to None.
            default_year: The default selection for which year's results to show. Defaults to None,
                causing it to use a system-wide default.
            default_coverage: The default selection for coverage level. Defaults to None, causing
                it to use a system-wide default.
            unit: The type of unit / unit size for which to show results. Defaults to unit risk
                which is the result of simulating with actual historic unit sizes.
            comparison: Which simulation to use as a baseline against which to compare simulation
                results. Defaults to 'vs counterfact' which is the simulation of results into the
                future year assuming climate change stops.
            output_loc: Where to write this visualization if provided. If None, this visualization
                will run interactively. Defaults to None.
        """
        if output_loc:
            self._sketch = sketchingpy.Sketch2DStatic(
                SUB_CHART_WIDTH + 80 + 11,
                SUB_CHART_HEIGHT * 2 + 50 + 25 * 2 + 41
            )
        else:
            self._sketch = sketchingpy.Sketch2D(
                SUB_CHART_WIDTH + 80 + 11,
                SUB_CHART_HEIGHT * 2 + 50 + 25 * 2 + 41,
                target,
                loading_id
            )

        if csv_loc is None:
            csv_loc = './data/sim_hist.csv'

        if default_year is None:
            default_year = 2050

        if default_coverage is None:
            default_coverage = '75'

        self._csv_loc = csv_loc

        self._cached_raw = self._sketch.get_data_layer().get_csv(self._csv_loc)

        self._redraw_required = True
        self._click_waiting = False
        self._key_waiting = None
        self._last_mouse_x = None
        self._last_mouse_y = None

        def set_mouse_clicked(mouse):
            self._click_waiting = True

        def set_key_waiting(button):
            self._key_waiting = button.get_name()

        top_button_y = 5
        bottom_button_y = SUB_CHART_HEIGHT * 2 + 50 + 25 * 2 + 2 + 10

        self._target_set = str(default_year)
        self._year_buttons = buttons.ToggleButtonSet(
            self._sketch,
            5,
            top_button_y,
            'Year',
            ['2030', '2050'],
            str(default_year),
            lambda x: self._change_year(x),
            keyboard_button='y'
        )

        self._target_threshold = default_coverage + '% cov'
        self._threshold_buttons = buttons.ToggleButtonSet(
            self._sketch,
            SUB_CHART_WIDTH + 80 - const.BUTTON_WIDTH * 2,
            top_button_y,
            'Threshold',
            ['85% cov', '75% cov'],
            default_coverage + '% cov',
            lambda x: self._change_loss(x),
            keyboard_button='c'
        )

        self._comparison = comparison
        self._comparison_buttons = buttons.ToggleButtonSet(
            self._sketch,
            5,
            bottom_button_y,
            'Comparison',
            ['vs counterfact', 'vs historic'],
            str(self._comparison),
            lambda x: self._change_comparison(x),
            keyboard_button='v'
        )

        self._geohash_size = unit
        self._geohash_buttons = buttons.ToggleButtonSet(
            self._sketch,
            SUB_CHART_WIDTH + 80 - const.BUTTON_WIDTH * 2,
            bottom_button_y,
            'Geohash',
            ['unit risk', 'sub-unit risk'],
            str(self._geohash_size),
            lambda x: self._change_geohash_size(x),
            keyboard_button='u'
        )

        self._records = self._get_records()

        if output_loc:
            self.draw()
            self._sketch.save_image(output_loc)
        else:
            mouse = self._sketch.get_mouse()
            mouse.on_button_press(set_mouse_clicked)

            keyboard = self._sketch.get_keyboard()
            keyboard.on_key_press(set_key_waiting)

            self._sketch.set_fps(10)
            self._sketch.on_step(lambda x: self.draw())
            self._sketch.show()

    def draw(self):
        """Update this visualization (execute a draw loop)."""
        mouse = self._sketch.get_mouse()

        if mouse is not None:
            mouse_x = mouse.get_pointer_x()
            mouse_y = mouse.get_pointer_y()
        else:
            mouse_x = 0
            mouse_y = 0

        mouse_x_changed = self._last_mouse_x != mouse_x
        mouse_y_changed = self._last_mouse_y != mouse_y
        mouse_changed = mouse_x_changed or mouse_y_changed
        change_waiting = self._click_waiting or self._redraw_required or self._key_waiting
        draw_skippable = not (change_waiting or mouse_changed)
        if draw_skippable:
            return

        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.clear(const.BACKGROUND_COLOR)
        self._draw_viz()

        self._year_buttons.step(mouse_x, mouse_y, self._click_waiting, self._key_waiting)
        self._threshold_buttons.step(mouse_x, mouse_y, self._click_waiting, self._key_waiting)
        self._comparison_buttons.step(mouse_x, mouse_y, self._click_waiting, self._key_waiting)
        self._geohash_buttons.step(mouse_x, mouse_y, self._click_waiting, self._key_waiting)

        self._click_waiting = False
        self._key_waiting = None

        self._sketch.pop_style()
        self._sketch.pop_transform()

        self._last_mouse_x = mouse_x
        self._last_mouse_y = mouse_y

    def _change_year(self, year_str, update_buttons=False):
        """Respond to the user changing the year in the visualization.

        Args:
            year_str: The year string as shown in the year buttons.
            update_buttons: Flag indicating if the UI buttons should be updated. True if the value
                should be updated and false if only internal state should change. Defaults to
                false.
        """
        self._target_set = year_str
        self._records = self._get_records()
        self._redraw_required = True

        if update_buttons:
            self._year_buttons.set_value(year_str)

    def _change_loss(self, loss_str, update_buttons=False):
        """Respond to the user changing the loss threshold in the visualization.

        Args:
            loss_str: The loss string as shown in the threshold buttons (85% cov, etc).
            update_buttons: Flag indicating if the UI buttons should be updated. True if the value
                should be updated and false if only internal state should change. Defaults to
                false.
        """
        self._target_threshold = loss_str
        self._records = self._get_records()
        self._redraw_required = True

        if update_buttons:
            self._threshold_buttons.set_value(loss_str)

    def _change_comparison(self, comparison_str, update_buttons=False):
        """Respond to the user changing the reference histogram in the visualization.

        Args:
            comparison_str: The year string as shown in the comparison buttons (vs counterfact,
                etc).
            update_buttons: Flag indicating if the UI buttons should be updated. True if the value
                should be updated and false if only internal state should change. Defaults to
                false.
        """
        self._comparison = comparison_str
        self._records = self._get_records()
        self._redraw_required = True

        if update_buttons:
            self._comparison_buttons.set_value(comparison_str)

    def _change_geohash_size(self, geohash_str, update_buttons=False):
        """Change the geohash size selected by the user.

        Args:
            geohash_str: The geohash and unit size to use (unit risk, etc).
            update_buttons: Flag indicating if the UI buttons should be updated. True if the value
                should be updated and false if only internal state should change. Defaults to
                false.
        """
        self._geohash_size = geohash_str
        self._records = self._get_records()
        self._redraw_required = True

        if update_buttons:
            self._geohash_buttons.set_value(geohash_str)

    def _get_x(self, value):
        """Get the x position of a histogram bucket.

        Args:
            value: The yield change to position horizontally.

        Returns:
            The x coordinate for the value in pixels.
        """
        offset = value + 105
        return offset / 200 * SUB_CHART_WIDTH

    def _get_y(self, value):
        """Get the y position of a histogram frequency.

        Args:
            value: The percent that falls within a histogram bucket.

        Returns:
            The y coordinate for the value in pixels.
        """
        return value / 15 * (SUB_CHART_HEIGHT - 20)

    def _combine_dicts(self, a, b):
        """Combine two dictionaries with numeric values.

        Combine two dictionaries by combining their key sets and adding values when appearing in
        both.

        Args:
            a: The first dictionary to combine.
            b: The second dictionary to combine.

        Returns:
            The result of combining the two dictionaries together.
        """

        def combine_inner(a_inner, b_inner):
            if a_inner is None:
                return b_inner
            elif b_inner is None:
                return a_inner
            else:
                keys = set(a_inner.keys()).union(set(b_inner.keys()))
                return dict(map(
                    lambda key: (key, a_inner.get(key, 0) + b_inner.get(key, 0)),
                    keys
                ))

        keys = set(a.keys()).union(set(b.keys()))
        return dict(map(
            lambda key: (key, combine_inner(a.get(key, None), b.get(key, None))),
            keys
        ))

    def _get_percents(self, target, claims_key):
        """Calculate the claims rate and percent of total for input summary records.

        Args:
            target: Mapping from series name to a dict mapping bin to count.
            claims_key: The type of claim as a string (claimsSco, claimsMpci).

        Returns:
            Dictionary mapping from series to record with counts converted to percentages and a
            claims rate.
        """

        def get_precent_inner(inner_target):
            keys = inner_target.keys()
            keys_allowed = list(filter(lambda x: x not in SUMMARY_FIELDS, keys))
            total = sum(map(lambda key: inner_target[key], keys_allowed))
            ret_dict = dict(map(
                lambda key: (key, inner_target[key] / total * 100),
                keys_allowed
            ))

            for key in SUMMARY_FIELDS:
                ret_dict[key] = inner_target.get(key, -1)

            ret_dict['claimsRate'] = float(inner_target[claims_key]) / float(inner_target['cnt'])

            return ret_dict

        keys = target.keys()
        ret_tuples = map(lambda key: (key, target[key]), keys)
        ret_tuples_transform = map(
            lambda x: (x[0], get_precent_inner(x[1])),
            ret_tuples
        )
        return dict(ret_tuples_transform)

    def _interpret_bin(self, target):
        """Interpret the name of a histogram bin or summary field.

        Args:
            target: The name of the bin which may be a summary field.

        Returns:
            The bin as a number corresponding to the yield change or the name of the summary field.
        """
        if target in SUMMARY_FIELDS:
            return target
        else:
            return int(target)

    def _get_records(self):
        """Parse all records from the cached raw copy.

        Returns:
            Parsed / loaded records.
        """
        raw_records = self._cached_raw

        target_geohash_size = {
            'unit risk': 4,
            'sub-unit risk': 5
        }[self._geohash_size]

        raw_records_right_size = filter(
            lambda x: int(x['geohashSize']) == target_geohash_size,
            raw_records
        )

        use_historic = self._comparison == 'vs historic'

        def get_is_in_target_series(target):
            is_target_set = target['set'] == self._target_set
            if use_historic:
                if target['set'] == '2010':
                    return True
                else:
                    return target['series'] == 'predicted' and is_target_set
            else:
                return target['series'] in SERIES and is_target_set

        allowed_records = filter(
            get_is_in_target_series,
            raw_records_right_size
        )

        def cast_individual_record(x):
            force_counterfactual = use_historic and x['set'] == '2010'
            return {
                'series': 'counterfactual' if force_counterfactual else x['series'],
                'bin': self._interpret_bin(x['bin']),
                'val': float(x['val'])
            }

        cast_records = map(
            cast_individual_record,
            allowed_records
        )

        nested_records = map(
            lambda x: {x['series']: {x['bin']: x['val']}},
            cast_records
        )

        counts = functools.reduce(lambda a, b: self._combine_dicts(a, b), nested_records)

        claims_key = 'claimsSco' if self._target_threshold == '85% cov' else 'claimsMpci'

        return self._get_percents(counts, claims_key)

    def _get_body_fields(self, hist):
        """Get all non-summary fields from the histogram dataset.

        Args:
            hist: The dictionary representing the outputs in histogram format.

        Returns:
            Items from the input dictionary that are not summary fields.
        """
        items = hist.items()
        allowed_items = filter(lambda x: x[0] not in SUMMARY_FIELDS, items)
        return allowed_items

    def _draw_upper(self, histogram):
        """Draw the distribution from the experimental set.

        Args:
            histogram: The results to draw on the upper part of the visualization.
        """
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.translate(50, 20)
        self._sketch.clear_stroke()
        self._sketch.set_fill(TOP_COLOR)
        self._sketch.set_rect_mode('corner')
        self._sketch.set_text_align('center', 'center')
        self._sketch.set_angle_mode('degrees')
        self._sketch.set_text_font(const.FONT_SRC, 11)

        is_catastrophic = self._target_threshold == '75% cov'
        claim_threshold = -25 if is_catastrophic else -15

        for bucket, percent in self._get_body_fields(histogram):
            is_claim = bucket <= claim_threshold

            if is_claim:
                self._sketch.clear_stroke()
                self._sketch.set_fill(TOP_COLOR)
            else:
                self._sketch.set_stroke(TOP_COLOR)
                self._sketch.set_stroke_weight(2)
                self._sketch.clear_fill()

            x = self._get_x(bucket)
            height = self._get_y(percent)
            self._sketch.draw_rect(
                x - 5,
                SUB_CHART_HEIGHT - height,
                10,
                height
            )

            if is_claim:
                self._sketch.set_text_font(const.FONT_SRC, 9)

                self._sketch.push_transform()
                self._sketch.translate(x, SUB_CHART_HEIGHT - height - 17)
                self._sketch.rotate(-90)
                self._sketch.draw_text(0, 0, '%.1f%%' % percent)
                self._sketch.pop_transform()

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _draw_lower(self, histogram):
        """Draw the distribution from the comparison set.

        Args:
            histogram: The results to draw on the lower part of the visualization.
        """
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.translate(50, 20 + SUB_CHART_HEIGHT + 50)
        self._sketch.set_rect_mode('corner')
        self._sketch.set_text_align('center', 'center')
        self._sketch.set_angle_mode('degrees')
        self._sketch.set_text_font(const.FONT_SRC, 11)

        is_catastrophic = self._target_threshold == '75% cov'
        claim_threshold = -25 if is_catastrophic else -15

        for bucket, percent in self._get_body_fields(histogram):
            is_claim = bucket <= claim_threshold
            if is_claim:
                self._sketch.clear_stroke()
                self._sketch.set_fill(BOTTOM_COLOR)
            else:
                self._sketch.set_stroke(BOTTOM_COLOR)
                self._sketch.set_stroke_weight(2)
                self._sketch.clear_fill()

            x = self._get_x(bucket)
            height = self._get_y(percent)
            self._sketch.draw_rect(
                x - 5,
                0,
                10,
                height
            )

            if is_claim:
                self._sketch.set_text_font(const.FONT_SRC, 9)

                self._sketch.push_transform()
                self._sketch.translate(x, height + 15)
                self._sketch.rotate(-90)
                self._sketch.draw_text(0, 0, '%.1f%%' % percent)
                self._sketch.pop_transform()

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _draw_x_axis(self, top_mean, bottom_mean):
        """Draw the shared x axis displaying changes in yield.

        Args:
            top_mean: The average of the top or experimental results.
            bottom_mean: The average of the bottom or comparison results.
        """
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.translate(50, 20 + SUB_CHART_HEIGHT + 25)
        self._sketch.clear_stroke()
        self._sketch.set_fill(TOP_COLOR)
        self._sketch.set_text_align('center', 'center')
        self._sketch.set_text_font(const.FONT_SRC, 13)
        self._sketch.set_ellipse_mode('radius')

        for bucket in range(-100, 120, 20):
            base_str = '+%d%%' % bucket if bucket > 0 else '%d%%' % bucket

            if bucket == 100:
                label = '>' + base_str
            else:
                label = base_str

            self._sketch.draw_text(
                self._get_x(bucket),
                0,
                label
            )

        self._sketch.set_text_align('center', 'top')
        self._sketch.set_text_font(const.FONT_SRC, 11)
        self._sketch.draw_text(
            self._get_x(80),
            7,
            'yield increase'
        )
        self._sketch.draw_text(
            self._get_x(-80),
            7,
            'yield decrease'
        )

        self._sketch.set_text_align('left', 'center')
        self._sketch.set_text_font(const.FONT_SRC, 11)

        self._sketch.set_fill(TOP_COLOR)
        self._sketch.clear_stroke()
        self._sketch.draw_ellipse(
            self._get_x(top_mean),
            -17,
            3,
            3
        )

        self._sketch.draw_text(
            self._get_x(top_mean) + 5,
            -17,
            ('+' if top_mean > 0 else '') + ('%.0f%% Mean' % top_mean)
        )

        self._sketch.set_fill(BOTTOM_COLOR)
        self._sketch.clear_stroke()
        self._sketch.draw_ellipse(
            self._get_x(bottom_mean),
            17,
            3,
            3
        )

        self._sketch.draw_text(
            self._get_x(bottom_mean) + 5,
            17,
            ('+' if bottom_mean > 0 else '') + ('%.0f%% Mean' % bottom_mean)
        )

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _draw_axis_bottom(self):
        """Draw the vertical axis for the bottom downward-facing histogram."""
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.translate(45, 20 + SUB_CHART_HEIGHT + 50)
        self._sketch.clear_stroke()
        self._sketch.set_fill(BOTTOM_COLOR)
        self._sketch.set_text_align('right', 'center')
        self._sketch.set_text_font(const.FONT_SRC, 13)

        for percent in range(0, 20, 5):
            height = self._get_y(percent)
            self._sketch.draw_text(
                5,
                height,
                '%d%%' % round(percent)
            )

        self._sketch.push_transform()
        self._sketch.translate(-35, self._get_y(7))
        self._sketch.set_angle_mode('degrees')
        self._sketch.rotate(-90)
        self._sketch.set_text_align('center', 'center')

        if self._comparison == 'vs historic':
            self._sketch.draw_text(0, 0, 'Predicted 2010 Series')
        else:
            self._sketch.draw_text(0, 0, 'Climate Change Stops')

        self._sketch.pop_transform()

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _draw_axis_upper(self):
        """Draw the vertical axis for the upper upwards-facing histogram."""
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.translate(45, 20)
        self._sketch.clear_stroke()
        self._sketch.set_fill(TOP_COLOR)
        self._sketch.set_text_align('right', 'center')
        self._sketch.set_text_font(const.FONT_SRC, 13)

        for percent in range(0, 20, 5):
            height = self._get_y(percent)
            self._sketch.draw_text(
                5,
                SUB_CHART_HEIGHT - height,
                '%d%%' % round(percent)
            )

        self._sketch.set_text_align('left', 'center')
        self._sketch.draw_text(
            8,
            SUB_CHART_HEIGHT - self._get_y(10),
            'of risk units'
        )

        self._sketch.push_transform()
        self._sketch.translate(-35, SUB_CHART_HEIGHT - self._get_y(7))
        self._sketch.set_angle_mode('degrees')
        self._sketch.rotate(-90)
        self._sketch.set_text_align('center', 'center')
        self._sketch.draw_text(0, 0, 'Continued Climate Change (SSP245)')
        self._sketch.pop_transform()

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _draw_top_claims(self, claims):
        """Draw the embedded bar chart with the claims on the top of the chart.

        Args:
            claims: The claims rate for the experimental results.
        """
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.translate(50, 20)
        self._sketch.clear_stroke()
        self._sketch.set_rect_mode('corner')
        self._sketch.set_text_align('center', 'center')
        self._sketch.set_text_font(const.FONT_SRC, 11)

        is_catastrophic = self._target_threshold == '75% cov'
        max_val = 35 if is_catastrophic else 45

        y = SUB_CHART_HEIGHT - self._get_y(15)
        start_x = self._get_x(-100) - 5
        end_x = self._get_x(-25 if is_catastrophic else -15)

        self._sketch.set_stroke(TOP_COLOR)
        self._sketch.set_stroke_weight(1)
        self._sketch.draw_line(start_x, y, end_x, y)

        self._sketch.clear_stroke()
        self._sketch.set_fill(TOP_COLOR)
        width = claims / max_val * (end_x - start_x)
        self._sketch.draw_rect(
            end_x - width,
            y + 2,
            width,
            5
        )

        self._sketch.set_text_align('right', 'top')
        self._sketch.draw_text(end_x, y + 8, '%.1f%% Loss Probability (Claims Rate)' % claims)

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _draw_bottom_claims(self, claims):
        """Draw the embedded bar chart with the claims on the bottom of the chart.

        Args:
            claims: The claims rate for the control or reference results.
        """
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.translate(50, 20 + SUB_CHART_HEIGHT + 50)
        self._sketch.clear_stroke()
        self._sketch.set_rect_mode('corner')
        self._sketch.set_text_font(const.FONT_SRC, 11)

        is_catastrophic = self._target_threshold == '75% cov'
        max_val = 35 if is_catastrophic else 45

        y = self._get_y(15)
        start_x = self._get_x(-100)
        end_x = self._get_x(-25 if is_catastrophic else -15)

        self._sketch.set_stroke(BOTTOM_COLOR)
        self._sketch.set_stroke_weight(1)
        self._sketch.draw_line(start_x, y, end_x, y)

        self._sketch.clear_stroke()
        self._sketch.set_fill(BOTTOM_COLOR)
        width = claims / max_val * (end_x - start_x)
        self._sketch.draw_rect(
            end_x - width,
            y - 7,
            width,
            5
        )

        self._sketch.set_text_align('right', 'bottom')
        self._sketch.draw_text(end_x, y - 8, '%.1f%% Loss Probability (Claims Rate)' % claims)

        self._sketch.set_text_align('right', 'top')
        self._sketch.draw_text(end_x, y + 2, '0%')

        self._sketch.set_text_align('left', 'top')
        self._sketch.draw_text(start_x, y + 2, '%d%%' % max_val)

        self._sketch.set_text_align('center', 'top')
        self._sketch.draw_text((start_x + end_x) / 2, y + 2, 'Loss Probability')

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _draw_title(self):
        """Draw the overall title at the top of the chart."""
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.clear_stroke()
        self._sketch.set_fill(TOP_COLOR)
        self._sketch.set_text_font(const.FONT_SRC, 16)
        self._sketch.set_text_align('center', 'bottom')
        self._sketch.draw_text(
            SUB_CHART_WIDTH / 2 + 40,
            25,
            'Histogram of Change in Risk Unit-Level Yields Relative to Expected (Avg Yield)'
        )

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _draw_viz(self):
        """Draw the body of the visualization."""
        self._sketch.push_transform()
        self._sketch.push_style()

        if self._redraw_required:
            self._sketch.create_buffer(
                'hist-center',
                SUB_CHART_WIDTH + 80,
                SUB_CHART_HEIGHT * 2 + 50 + 20,
                '#FFFFFF'
            )

            self._sketch.enter_buffer('hist-center')
            self._sketch.clear('#FFFFFF')
            self._draw_upper(self._records['predicted'])
            self._draw_lower(self._records['counterfactual'])
            self._draw_x_axis(
                self._records['predicted']['mean'] * 100,
                self._records['counterfactual']['mean'] * 100
            )
            self._draw_axis_upper()
            self._draw_axis_bottom()
            self._draw_top_claims(self._records['predicted']['claimsRate'] * 100)
            self._draw_bottom_claims(self._records['counterfactual']['claimsRate'] * 100)
            self._draw_title()
            self._sketch.exit_buffer()

            self._redraw_required = False

        self._sketch.clear_fill()
        self._sketch.set_stroke(const.INACTIVE_BORDER)
        self._sketch.set_rect_mode('corner')
        self._sketch.draw_rect(
            4,
            4 + 25 + 5,
            SUB_CHART_WIDTH + 80 + 2,
            SUB_CHART_HEIGHT * 2 + 50 + 20 + 2
        )
        self._sketch.draw_buffer(5, 5 + 25 + 5, 'hist-center')

        self._sketch.pop_style()
        self._sketch.pop_transform()


def main():
    """Main entrypoint for this visualization.

    Main entrypoint for this visualization, executing interactively if not command line arguments.
    Otherwise, will write to file and run headless.
    """
    if len(sys.argv) == 1:
        presenter = MainPresenter('Simulation Outcomes', None)
    elif len(sys.argv) != NUM_ARGS + 1:
        print(USAGE_STR)
        sys.exit(1)
    else:
        csv_loc = sys.argv[1]
        year = sys.argv[2]
        coverage = sys.argv[3]
        risk_unit = sys.argv[4]
        comparison = sys.argv[5]
        output_loc = sys.argv[6]

        presenter = MainPresenter(
            'Simulation Outcomes',
            None,
            csv_loc=csv_loc,
            default_year=year,
            default_coverage=coverage,
            output_loc=output_loc,
            unit='unit risk' if risk_unit == 'unit' else 'sub-unit risk',
            comparison='vs counterfact' if comparison == 'counterfactual' else 'vs historic'
        )

    assert presenter is not None


if __name__ == '__main__':
    main()
