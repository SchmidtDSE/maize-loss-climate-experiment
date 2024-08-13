"""The main presenter for the results visualization that coordinates other components.

The main presenter for the results visualization (neighborhood visualiation) that coordinates other
components.

License:
BSD
"""

import sketchingpy

import buttons
import config_buttons
import const
import data_struct
import legend
import map_viz
import preprocess
import scatter


class ResultsVizPresenter:
    """Main presenter for the neighborhood-level results viz that coordinates other components."""

    def __init__(self, target, loading_id, default_configuration=None, data_loc=None,
        climate_loc=None, output_loc=None):
        """Create a enw results viz instance.
        
        Args:
            target: The ID in which the visualization should be built or the title of the window if
                not running the browser.
            loading_id: The ID of the loading indicator that should be hidden after initalization.
            default_configuration: The default configuration (Configuration) to use as the initial
                configuration of the tool.
            data_loc: The location where the neighborhood-level results CSV can be found or None if
                a system-wide default should be used. Defaults to None.
            climate_loc: The location where neighborhood-level climate changes CSV can be found or
                None if a system-wide default should be used. Defaults to None.
            output_loc: The location at which the visualization should be written. If None, will
                run interactively. Defaults to None.
        """

        if output_loc:
            self._sketch = sketchingpy.Sketch2DStatic(const.WIDTH, const.HEIGHT)
        else:
            self._sketch = sketchingpy.Sketch2D(const.WIDTH, const.HEIGHT, target, loading_id)

        if data_loc:
            self._data_loc = data_loc
        else:
            self._data_loc = 'data/tool.csv'

        if climate_loc:
            self._climate_loc = climate_loc
        else:
            self._climate_loc = 'data/climate.csv'

        self._sketch.set_fps(10)
        self._all_records = self._load_records()
        self._climate_deltas = self._load_climate_deltas()
        self._selected_geohashes = set()
        self._selecting = False

        self._last_mouse_x = None
        self._last_mouse_y = None
        self._change_waiting = False

        main_height = const.HEIGHT - 100

        if default_configuration:
            self._config = default_configuration
        else:
            self._config = config_buttons.Configuration(
                '2050 series',
                'Avg All Years',
                'yield',
                'scatter',
                'p <  0.05',
                'Bonferroni',
                'significant only',
                'no var',
                'jul',
                '75% cov'
            )

        data_initial = preprocess.make_scatter_values(
            self._all_records,
            self._climate_deltas,
            self._config
        )

        self._select_button = buttons.Button(
            self._sketch,
            5,
            10,
            'Select Fields',
            lambda x: self._start_fields_selection()
        )

        self._scatter_presenter = scatter.ScatterMainPresenter(
            self._sketch,
            5,
            50,
            const.MAIN_WIDTH,
            main_height,
            data_initial,
            self._config.get_metric(),
            self._config.get_var(),
            self._selected_geohashes,
            lambda x: self._update_selected_geohashes(x)
        )

        self._map_presenter = map_viz.MapMainPresenter(
            self._sketch,
            5,
            50,
            const.MAIN_WIDTH,
            main_height,
            data_initial,
            self._config.get_metric(),
            self._config.get_var(),
            self._selected_geohashes,
            lambda x: self._update_selected_geohashes(x)
        )

        percents_initial = preprocess.make_percents(data_initial)
        self._legend_presenter = legend.LegendPresenter(
            self._sketch,
            const.MAIN_WIDTH + 30,
            50 + main_height - 130,
            250,
            125,
            percents_initial,
            self._config.get_metric(),
            self._config.get_var(),
            self._config.get_visualization()
        )

        self._configuration_pesenter = config_buttons.ConfigurationPresenter(
            self._sketch,
            const.MAIN_WIDTH + 30,
            50,
            self._config,
            lambda x: self._change_config(x)
        )

        self._clicked = False
        self._key_waiting = None

        if output_loc:
            self._refresh_data()
            self._step()
            self._sketch.save_image(output_loc)
        else:
            mouse = self._sketch.get_mouse()

            def set_mouse_clicked(mouse):
                self._clicked = True

            mouse.on_button_press(set_mouse_clicked)

            keyboard = self._sketch.get_keyboard()

            def set_key_waiting(button):
                self._key_waiting = button.get_name()

            keyboard.on_key_press(set_key_waiting)

            self._sketch.on_step(lambda x: self._step())
            self._sketch.show()

    def _update_selected_geohashes(self, selected_geohashes):
        """Update collection of highlighted geohashes.
        
        Args:
            selected_geohashes: Collection of string geohashses highlighted by the user.
        """
        self._selected_geohashes = selected_geohashes
        self._refresh_data()

    def _change_config(self, new_config):
        """Change the configuration selected.
        
        Note:
            At this time, this does not force UI elements to update display.
        
        Args:
            new_config: The Configuration to show.
        """
        self._config = new_config
        self._refresh_data()

    def _refresh_data(self):
        """Reload data and make visualization supporting structures."""
        self._selecting = False

        self._change_waiting = True

        new_data = preprocess.make_scatter_values(
            self._all_records,
            self._climate_deltas,
            self._config
        )

        self._scatter_presenter.update_data(
            new_data,
            self._config.get_metric(),
            self._config.get_var(),
            self._selected_geohashes
        )

        self._map_presenter.update_data(
            new_data,
            self._config.get_metric(),
            self._config.get_var(),
            self._selected_geohashes
        )

        new_percents = preprocess.make_percents(new_data)
        self._legend_presenter.update_data(
            new_percents,
            self._config.get_metric(),
            self._config.get_var(),
            self._config.get_visualization()
        )

    def _step(self):
        """Update the visualization and display."""
        mouse = self._sketch.get_mouse()

        if mouse:
            mouse_x = mouse.get_pointer_x()
            mouse_y = mouse.get_pointer_y()
        else:
            mouse_x = 0
            mouse_y = 0

        mouse_x_same = mouse_x == self._last_mouse_x
        mouse_y_same = mouse_y == self._last_mouse_y
        event_waiting = (self._clicked) or (self._key_waiting is not None)
        mouse_same = mouse_x_same and mouse_y_same and (not event_waiting)
        if mouse_same and not self._change_waiting:
            return
        else:
            self._last_mouse_x = mouse_x
            self._last_mouse_y = mouse_y
            self._change_waiting = False

        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.clear(const.BACKGROUND_COLOR)

        if self._config.get_visualization() == 'scatter':
            self._scatter_presenter.step(mouse_x, mouse_y, self._clicked)
        else:
            self._map_presenter.step(mouse_x, mouse_y, self._clicked)

        self._configuration_pesenter.step(mouse_x, mouse_y, self._clicked, self._key_waiting)
        self._legend_presenter.step(mouse_x, mouse_y, self._clicked)

        if not self._selecting:
            self._select_button.step(mouse_x, mouse_y, self._clicked)

        self._draw_annotation()

        self._sketch.pop_style()
        self._sketch.pop_transform()

        self._clicked = False
        self._key_waiting = None

    def _start_fields_selection(self):
        """Enable the field highlighting mode."""
        self._scatter_presenter.start_selecting()
        self._map_presenter.start_selecting()
        self._selecting = True

    def _draw_annotation(self):
        """Draw instructional / status text."""
        self._sketch.push_transform()
        self._sketch.push_style()

        self._sketch.set_fill('#333333')
        self._sketch.clear_stroke()
        self._sketch.set_text_font(const.FONT_SRC, 12)
        self._sketch.set_text_align('left', 'center')
        self._sketch.draw_text(130, 17, const.SELECTION_INSTRUCTION)
        self._sketch.draw_text(130, 29, self._get_description())

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def _get_description(self):
        """Get a description of the current configuration of the visualization.
        
        Returns:
            String describing the current state of the visualization.
        """
        if self._config.get_risk_range() == '':
            agg_str = 'Averaging across all years.'
        else:
            agg_str = 'Single year avg yields.'

        in_map = self._config.get_visualization() == 'map'
        no_var = self._config.get_var() == 'no var'
        if in_map and not no_var:
            area_str = 'Color showing change in input var.'
        else:
            area_str = 'Area of circle proportional to num fields.'

        description = ' '.join([
            'Using CHC-CMIP6 and Lobell.',
            agg_str,
            area_str
        ])

        return description

    def _load_records(self):
        """Load the Records for this visualization.
        
        Returns:
            List of Record.
        """
        data_layer = self._sketch.get_data_layer()
        reader = data_layer.get_csv(self._data_loc)
        return [data_struct.parse_record(x) for x in reader]

    def _load_climate_deltas(self):
        """Load information about climate deltas for this visualization.
        
        Returns:
            List of ClimateDelta.
        """
        data_layer = self._sketch.get_data_layer()
        reader = data_layer.get_csv(self._climate_loc)
        records = map(lambda x: data_struct.parse_climate_delta(x), reader)
        deltas = data_struct.ClimateDeltas(records)
        return deltas
