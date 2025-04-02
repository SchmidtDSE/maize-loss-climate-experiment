"""Map component for the neighborhood-level results visualization.

License:
    BSD
"""

import math

import const
import data_struct
import symbols


class MapMainPresenter:
    """Map component presenter."""

    def __init__(self, sketch, x, y, width, height, records, metric, var, selected_geohashes,
        show_acreage, on_selection):
        """Create a new map component.

        Args:
            sketch: The Sketchingpy sketch in which the map should be created.
            x: The horizontal position where the map component should be constructed.
            y: The vertical position where the map component should be constructed.
            width: The horizontal size in pixels of the map area.
            height: The vertical size in pixels of the map area.
            records: The Records to be displayed within this map.
            metric: The metric (matching Configuration) like risk or yield being displayed.
            var: The contextualizing dimension.
            selected_geohashes: Set of geohashes currently highglighted by the user (collection of
                strings).
            show_acreage: Flag indicating if the acreage should be shown.
            on_selection: Callback to invoke when the collection of selected geohashes changes.
        """
        self._sketch = sketch
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._records = records
        self._metric = metric
        self._var = var

        self._selected_geohashes = selected_geohashes
        self._on_selection = on_selection
        self._show_acreage = show_acreage

        self._needs_redraw = False
        self._selecting = False
        self._placed_records = []

        self._make_map_image(records, metric, var)

    def start_selecting(self):
        """Indicate that the visualization has entered into the highlighting state."""
        self._selecting = True

    def step(self, mouse_x, mouse_y, clicked):
        """Update the map visualization and redraw.

        Args:
            mouse_x: The horizontal position of the cursor.
            mouse_y: The vertical position of the cursor.
            clicked: Flag indicating if the mouse button has been pressed since the last call to
                step. True if pressed and false otherwise.
        """
        self._sketch.push_transform()
        self._sketch.push_style()

        if self._needs_redraw:
            self._make_map_image(self._records, self._metric, self._var)
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

        self._sketch.draw_buffer(1, 1, 'map')

        hovering_x = mouse_x_offset > 0 and mouse_x_offset < self._width
        hovering_y = mouse_y_offset > 0 and mouse_y_offset < self._height
        hovering = hovering_x and hovering_y

        if hovering and self._selecting:
            self._sketch.clear_fill()
            self._sketch.set_stroke(const.SELECT_COLOR)
            self._sketch.set_stroke_weight(2)
            self._sketch.set_ellipse_mode('radius')
            self._sketch.draw_ellipse(mouse_x_offset, mouse_y_offset, 30, 30)

            if clicked:
                self._select_points(mouse_x_offset, mouse_y_offset)

        self._sketch.pop_style()
        self._sketch.pop_transform()

    def update_data(self, records, metric, var, selected_geohashes, show_acreage):
        """Update the data to display within the map.

        Args:
            records: The Records to display.
            metric: The metric like risk or yield to display within the map.
            var: The contextualizing dimension to display.
            selected_geohashes: Collection of geohashes highlighted by the user.
            show_acreage: Flag indicating if the acreage should be shown.
        """
        self._records = records
        self._metric = metric
        self._var = var
        self._selected_geohashes = selected_geohashes
        self._show_acreage = show_acreage
        self._needs_redraw = True
        self._selecting = False

    def _select_points(self, rel_x, rel_y):
        """Callback for when the user selects points within the map.

        Args:
            rel_x: The horizontal cursor position relative to the start of the map in pixels.
            rel_y: The vertical cursor position relative to the start of the map in pixels.
        """
        in_range = filter(lambda x: x.in_range(rel_x, rel_y, 30), self._placed_records)
        geohashes = set(map(lambda x: x.get_record().get_geohash(), in_range))
        self._on_selection(geohashes)
        self._selecting = False

    def _make_map_image(self, records, metric, var):
        """Make a cached (buffered) map image.

        Args:
            records: The Records to display within the map image.
            metric: The metric like risk or yield to display within the map.
            var: The contextualizing dimension to display.
        """
        usable_height = self._get_usable_height() - 2
        usable_width = self._width - 2

        self._sketch.create_buffer('map', usable_width, usable_height, '#F8F8F8')

        self._sketch.push_transform()
        self._sketch.push_style()
        self._sketch.enter_buffer('map')

        self._sketch.set_map_pan(-91.8, 42.5)
        self._sketch.set_map_zoom(2.4)
        self._sketch.set_map_placement(self._width / 2, usable_height / 2)

        self._sketch.set_ellipse_mode('radius')
        self._sketch.set_rect_mode('radius')

        data_layer = self._sketch.get_data_layer()
        geojson = data_layer.get_json('data/states.json.geojson')
        geo_polygons = self._sketch.parse_geojson(geojson)

        self._sketch.set_fill('#FFFFFF')
        self._sketch.set_stroke_weight(1)
        self._sketch.set_stroke('#E0E0E0')

        for geo_polygon in geo_polygons:
            shape = geo_polygon.to_shape()
            self._sketch.draw_shape(shape)

        max_count = const.MAX_COUNT
        total = sum(map(lambda x: x.get_count(), records))

        def get_radius(r):
            if not self._show_acreage:
                return 2

            result = math.sqrt(r / max_count * 900)
            if result < 1:
                return 1
            else:
                return result

        def draw_record_no_var(record):
            if record.get_category() == 'no significant change':
                self._sketch.set_stroke_weight(1)
                self._sketch.set_stroke(const.CATEGORY_COLORS[record.get_category()] + 'A0')
                self._sketch.clear_fill()
            else:
                self._sketch.set_stroke_weight(1)
                self._sketch.set_stroke('#C0C0C0')
                self._sketch.set_fill(const.CATEGORY_COLORS[record.get_category()] + 'C0')

            if record.get_geohash() in self._selected_geohashes:
                self._sketch.set_stroke_weight(1)
                self._sketch.set_stroke(const.SELECTED_COLOR)

            longitude = record.get_longitude()
            latitude = record.get_latitude()
            x, y = self._sketch.convert_geo_to_pixel(longitude, latitude)
            r = get_radius(record.get_count() / total)
            self._sketch.draw_ellipse(x, y, r, r)
            self._placed_records.append(data_struct.PlacedRecord(x, y, record))

        def draw_record_var(record):
            longitude = record.get_longitude()
            latitude = record.get_latitude()
            x, y = self._sketch.convert_geo_to_pixel(longitude, latitude)
            strategy = symbols.get_strategy(self._sketch, record.get_category())

            if record.get_x_value() > 0:
                color_set = const.MAP_SCALE_POSITIVE
            else:
                color_set = const.MAP_SCALE_NEGATIVE

            min_val = const.VAR_MINS[var]
            max_val = const.VAR_MAXS[var]
            extent_val = max([abs(min_val), abs(max_val)])
            index = round(abs(record.get_x_value()) / extent_val * 3)
            if index > 3:
                index = 3

            self._sketch.set_fill(color_set[index])
            self._sketch.clear_stroke()
            self._sketch.draw_rect(x, y, 4, 4)

            if record.get_geohash() in self._selected_geohashes:
                color = '#333333' if index < 2 else '#FFFFFF'
            else:
                color = '#A0A0A0' if index < 2 else '#D0D0D0'

            strategy(x, y, color)
            self._placed_records.append(data_struct.PlacedRecord(x, y, record))

        if var == 'no var':
            draw_record = draw_record_no_var
        else:
            draw_record = draw_record_var

        del self._placed_records[:]
        for record in records:
            draw_record(record)

        self._sketch.pop_style()
        self._sketch.pop_transform()
        self._sketch.exit_buffer()

    def _get_usable_height(self):
        """Get the actual map height.

        Returns:
            Height of the usable map area in pixels.
        """
        return self._height - 5
