import functools
import sys

import sketchingpy

import buttons
import const

SERIES = ['predicted', 'counterfactual']
SUB_CHART_WIDTH = 700
SUB_CHART_HEIGHT = 200
TOP_COLOR = '#404040'
BOTTOM_COLOR = '#707070'
SUMMARY_FIELDS = ['claimsMpci', 'claimsSco', 'mean', 'cnt', 'claimsRate']

NUM_ARGS = 4
USAGE_STR = 'python hist_viz.py [csv location] [default year] [default coverage] [output location]'


class MainPresenter:

    def __init__(self, target, loading_id, csv_loc=None, default_year=None, default_coverage=None,
        output_loc=None):
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
            default_year = 2030

        if default_coverage is None:
            default_coverage = '75'

        self._csv_loc = csv_loc

        self._cached_raw = self._sketch.get_data_layer().get_csv(self._csv_loc)

        self._redraw_required = True
        self._click_waiting = False
        self._last_mouse_x = None
        self._last_mouse_y = None

        def set_mouse_clicked(mouse):
            self._click_waiting = True

        top_button_y = 5
        bottom_button_y = SUB_CHART_HEIGHT * 2 + 50 + 25 * 2 + 2 + 8

        self._target_set = str(default_year)
        self._year_buttons = buttons.ToggleButtonSet(
            self._sketch,
            5,
            top_button_y,
            'Year',
            ['2030', '2050'],
            str(default_year),
            lambda x: self._change_year(x)
        )

        self._target_threshold = default_coverage + '% cov'
        self._threshold_buttons = buttons.ToggleButtonSet(
            self._sketch,
            SUB_CHART_WIDTH + 80 - const.BUTTON_WIDTH * 2,
            top_button_y,
            'Threshold',
            ['85% cov', '75% cov'],
            default_coverage + '% cov',
            lambda x: self._change_loss(x)
        )

        self._comparison = 'vs counterfact'
        self._comparison_buttons = buttons.ToggleButtonSet(
            self._sketch,
            5,
            bottom_button_y,
            'Comparison',
            ['vs counterfact', 'vs historical'],
            str(self._comparison),
            lambda x: self._change_comparison(x)
        )

        self._geohash_size = '4 char geohash'
        self._geohash_buttons = buttons.ToggleButtonSet(
            self._sketch,
            SUB_CHART_WIDTH + 80 - const.BUTTON_WIDTH * 2,
            bottom_button_y,
            'Geohash',
            ['4 char geohash', 'approx 5 char'],
            str(self._geohash_size),
            lambda x: self._change_geohash_size(x)
        )

        self._records = self._get_records()

        if output_loc:
            self.draw()
            self._sketch.save_image(output_loc)
        else:
            mouse = self._sketch.get_mouse()
            mouse.on_button_press(set_mouse_clicked)

            self._sketch.set_fps(10)
            self._sketch.on_step(lambda x: self.draw())
            self._sketch.show()

    def draw(self):
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
        change_waiting = self._click_waiting or self._redraw_required
        draw_skippable = not (change_waiting or mouse_changed)
        if draw_skippable:
            return

        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.clear(const.BACKGROUND_COLOR)
        self._draw_viz()

        self._year_buttons.step(mouse_x, mouse_y, self._click_waiting)
        self._threshold_buttons.step(mouse_x, mouse_y, self._click_waiting)
        self._comparison_buttons.step(mouse_x, mouse_y, self._click_waiting)
        self._geohash_buttons.step(mouse_x, mouse_y, self._click_waiting)

        self._click_waiting = False

        self._sketch.pop_style()
        self._sketch.pop_transform()

        self._last_mouse_x = mouse_x
        self._last_mouse_y = mouse_y

    def _change_year(self, year_str):
        self._target_set = year_str
        self._records = self._get_records()
        self._redraw_required = True

    def _change_loss(self, loss_str):
        self._target_threshold = loss_str
        self._records = self._get_records()
        self._redraw_required = True

    def _change_comparison(self, comparison_str):
        self._comparison = comparison_str
        self._records = self._get_records()
        self._redraw_required = True

    def _change_geohash_size(self, geohash_str):
        self._geohash_size = geohash_str
        self._records = self._get_records()
        self._redraw_required = True

    def _get_x(self, value):
        offset = value + 100
        return offset / 200 * SUB_CHART_WIDTH

    def _get_y(self, value):
        return value / 20 * (SUB_CHART_HEIGHT - 50)

    def _combine_dicts(self, a, b):
        
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
        if target in SUMMARY_FIELDS:
            return target
        else:
            return int(target)

    def _get_records(self):
        raw_records = self._cached_raw
        
        target_geohash_size = {
            '4 char geohash': 4,
            'approx 5 char': 5
        }[self._geohash_size]
        
        raw_records_right_size = filter(
            lambda x: int(x['geohashSize']) == target_geohash_size,
            raw_records
        )

        use_historic = self._comparison == 'vs historical'
        
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
        
        cast_records = map(
            lambda x: {
                'series': 'counterfactual' if (use_historic and x['set'] == '2010') else x['series'],
                'bin': self._interpret_bin(x['bin']),
                'val': float(x['val'])
            },
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
        items = hist.items()
        allowed_items = filter(lambda x: x[0] not in SUMMARY_FIELDS, items)
        return allowed_items

    def _draw_upper(self, histogram):
        self._sketch.push_transform()
        self._sketch.push_style()
        
        self._sketch.translate(50, 20)
        self._sketch.clear_stroke()
        self._sketch.set_fill(TOP_COLOR)
        self._sketch.set_rect_mode('corner')
        self._sketch.set_text_align('center', 'center')
        self._sketch.set_angle_mode('degrees')
        self._sketch.set_text_font(const.FONT_SRC, 9)

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
                self._sketch.push_transform()
                self._sketch.translate(x, SUB_CHART_HEIGHT - height - 15)
                self._sketch.rotate(-90)
                self._sketch.draw_text(0, 0, '%.1f%%' % percent)
                self._sketch.pop_transform()
        
        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _draw_lower(self, histogram):
        self._sketch.push_transform()
        self._sketch.push_style()
        
        self._sketch.translate(50, 20 + SUB_CHART_HEIGHT + 50)
        self._sketch.set_rect_mode('corner')
        self._sketch.set_text_align('center', 'center')
        self._sketch.set_angle_mode('degrees')
        self._sketch.set_text_font(const.FONT_SRC, 9)

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
                self._sketch.push_transform()
                self._sketch.translate(x, height + 15)
                self._sketch.rotate(-90)
                self._sketch.draw_text(0, 0, '%.1f%%' % percent)
                self._sketch.pop_transform()
        
        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _draw_x_axis(self, top_mean, bottom_mean):
        self._sketch.push_transform()
        self._sketch.push_style()
        
        self._sketch.translate(50, 20 + SUB_CHART_HEIGHT + 25)
        self._sketch.clear_stroke()
        self._sketch.set_fill(TOP_COLOR)
        self._sketch.set_text_align('center', 'center')
        self._sketch.set_text_font(const.FONT_SRC, 11)
        self._sketch.set_ellipse_mode('radius')
        
        for bucket in range(-100, 120, 20):
            base_str = '+%d%%' % bucket if bucket > 0 else '%d%%' % bucket
            
            if bucket == -100:
                label = '<' + base_str
            elif bucket == 100:
                label = '>' + base_str
            else:
                label = base_str
            
            self._sketch.draw_text(
                self._get_x(bucket),
                0,
                label
            )
        
        self._sketch.set_text_align('center', 'top')
        self._sketch.set_text_font(const.FONT_SRC, 9)
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
        self._sketch.set_text_font(const.FONT_SRC, 9)
        
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
        self._sketch.push_transform()
        self._sketch.push_style()
        
        self._sketch.translate(45, 20 + SUB_CHART_HEIGHT + 50)
        self._sketch.clear_stroke()
        self._sketch.set_fill(BOTTOM_COLOR)
        self._sketch.set_text_align('right', 'center')
        self._sketch.set_text_font(const.FONT_SRC, 11)
        
        for percent in range(0, 25, 5):
            height = self._get_y(percent)
            self._sketch.draw_text(
                0,
                height,
                '%d%%' % round(percent)
            )
        
        self._sketch.push_transform()
        self._sketch.translate(-30, self._get_y(8))
        self._sketch.set_angle_mode('degrees')
        self._sketch.rotate(-90)
        self._sketch.set_text_align('center', 'center')
        self._sketch.draw_text(0, 0, 'Without Climate Change')
        self._sketch.pop_transform()
        
        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _draw_axis_upper(self):
        self._sketch.push_transform()
        self._sketch.push_style()
        
        self._sketch.translate(45, 20)
        self._sketch.clear_stroke()
        self._sketch.set_fill(TOP_COLOR)
        self._sketch.set_text_align('right', 'center')
        self._sketch.set_text_font(const.FONT_SRC, 11)
        
        for percent in range(0, 25, 5):
            height = self._get_y(percent)
            self._sketch.draw_text(
                0,
                SUB_CHART_HEIGHT - height,
                '%d%%' % round(percent)
            )
        
        self._sketch.set_text_align('left', 'center')
        self._sketch.draw_text(
            2,
            SUB_CHART_HEIGHT - self._get_y(10),
            'of risk units'
        )
        
        self._sketch.push_transform()
        self._sketch.translate(-30, SUB_CHART_HEIGHT - self._get_y(8))
        self._sketch.set_angle_mode('degrees')
        self._sketch.rotate(-90)
        self._sketch.set_text_align('center', 'center')
        self._sketch.draw_text(0, 0, 'With Climate Change')
        self._sketch.pop_transform()
        
        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _draw_top_claims(self, claims):
        self._sketch.push_transform()
        self._sketch.push_style()
        
        self._sketch.translate(50, 20)
        self._sketch.clear_stroke()
        self._sketch.set_rect_mode('corner')
        self._sketch.set_text_align('center', 'center')
        self._sketch.set_text_font(const.FONT_SRC, 9)

        is_catastrophic = self._target_threshold == '75% cov'
        max_val = 20 if is_catastrophic else 25
        
        y = SUB_CHART_HEIGHT - self._get_y(20)
        start_x = self._get_x(-100)
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
        self._sketch.draw_text(end_x, y + 8, '%.0f%% Loss Probability' % claims)
        
        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _draw_bottom_claims(self, claims):
        self._sketch.push_transform()
        self._sketch.push_style()
        
        self._sketch.translate(50, 20 + SUB_CHART_HEIGHT + 50)
        self._sketch.clear_stroke()
        self._sketch.set_rect_mode('corner')
        self._sketch.set_text_font(const.FONT_SRC, 9)

        is_catastrophic = self._target_threshold == '75% cov'
        max_val = 20 if is_catastrophic else 25
        
        y = self._get_y(20)
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
        self._sketch.draw_text(end_x, y - 8, '%.0f%% Loss Probability' % claims)
        
        self._sketch.set_text_align('right', 'top')
        self._sketch.draw_text(end_x, y + 2, '0%')
        
        self._sketch.set_text_align('left', 'top')
        self._sketch.draw_text(start_x, y + 2, '%d%%' % max_val)
        
        self._sketch.set_text_align('center', 'top')
        self._sketch.draw_text((start_x + end_x) / 2, y + 2, 'Catastrophic Loss Probability')
        
        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _draw_title(self):
        self._sketch.push_transform()
        self._sketch.push_style()
        
        self._sketch.clear_stroke()
        self._sketch.set_fill(TOP_COLOR)
        self._sketch.set_text_font(const.FONT_SRC, 16)
        self._sketch.set_text_align('center', 'bottom')
        self._sketch.draw_text(
            SUB_CHART_WIDTH / 2 + 40,
            30,
            'Histogram of Change in Risk Unit-Level Yields Relative to Expected (Avg Yield)'
        )
        
        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _draw_viz(self):
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
    if len(sys.argv) == 1:
        presenter = MainPresenter('Simulation Outcomes', None)
    elif len(sys.argv) != NUM_ARGS + 1:
        print(USAGE_STR)
        sys.exit(1)
    else:
        csv_loc = sys.argv[1]
        year = sys.argv[2]
        coverage = sys.argv[3]
        output_loc = sys.argv[4]
        
        presenter = MainPresenter(
            'Simulation Outcomes',
            None,
            csv_loc=csv_loc,
            default_year=year,
            default_coverage=coverage,
            output_loc=output_loc
        )


if __name__ == '__main__':
    main()
