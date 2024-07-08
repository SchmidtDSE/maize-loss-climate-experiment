def draw_not_significant(sketch, x, y, color):
    sketch.push_transform()
    sketch.push_style()

    #sketch.set_rect_mode('radius')
    #sketch.set_fill(color)
    #sketch.clear_stroke()
    #sketch.draw_rect(x, y, 2, 1)

    sketch.pop_style()
    sketch.pop_transform()


def draw_increase(sketch, x, y, color):
    sketch.push_transform()
    sketch.push_style()

    sketch.set_ellipse_mode('radius')
    sketch.set_fill(color)
    sketch.clear_stroke()
    sketch.draw_ellipse(x, y, 2, 2)

    sketch.pop_style()
    sketch.pop_transform()


def draw_increase_secondary(sketch, x, y, color):
    sketch.push_transform()
    sketch.push_style()

    sketch.set_rect_mode('radius')
    sketch.set_fill(color)
    sketch.clear_stroke()
    sketch.draw_rect(x, y, 2, 2)

    sketch.pop_style()
    sketch.pop_transform()


def draw_decrease(sketch, x, y, color):
    sketch.push_transform()
    sketch.push_style()

    sketch.set_ellipse_mode('radius')
    sketch.clear_fill()
    sketch.set_stroke(color)
    sketch.draw_ellipse(x, y, 2, 2)

    sketch.pop_style()
    sketch.pop_transform()


def draw_decrease_secondary(sketch, x, y, color):
    sketch.push_transform()
    sketch.push_style()

    sketch.set_rect_mode('radius')
    sketch.clear_fill()
    sketch.set_stroke(color)
    sketch.draw_rect(x, y, 2, 2)

    sketch.pop_style()
    sketch.pop_transform()


def get_strategy(sketch, category):
    inner_not_sig = lambda x, y, color: draw_not_significant(sketch, x, y, color)
    inner_decrease_primary = lambda x, y, color: draw_decrease(sketch, x, y, color)
    inner_decrease_secondary = lambda x, y, color: draw_decrease_secondary(sketch, x, y, color)
    inner_increase_primary = lambda x, y, color: draw_increase(sketch, x, y, color)
    inner_increase_secondary = lambda x, y, color: draw_increase_secondary(sketch, x, y, color)
    var_draw_strategies = {
        'not significant': inner_not_sig,
        'decrease': inner_decrease_primary,
        'increase': inner_increase_primary,
        'increased risk, can adapt': inner_increase_primary,
        'increased risk, cant adapt': inner_increase_secondary,
        'decreased risk, can adapt': inner_decrease_primary,
        'decreased risk, cant adapt': inner_decrease_secondary,
        'increase risk, increase variability': inner_increase_primary,
        'increase risk, decrease variability': inner_increase_secondary,
        'decrease risk, increase variability': inner_decrease_primary,
        'decrease risk, decrease variability': inner_decrease_secondary
    }
    return var_draw_strategies[category]
