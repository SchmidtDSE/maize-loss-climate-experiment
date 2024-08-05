def draw_not_significant(sketch, x, y, color):
    sketch.push_transform()
    sketch.push_style()

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
        'no significant change': inner_not_sig,
        'lower than counterfactual': inner_decrease_primary,
        'higher than counterfactual': inner_increase_primary,
        'higher risk, can adapt': inner_increase_primary,
        'higher risk, cant adapt': inner_increase_secondary,
        'lower risk, can adapt': inner_decrease_primary,
        'lower risk, cant adapt': inner_decrease_secondary,
        'higher risk, yield above counterfactual': inner_increase_primary,
        'higher risk, yield below counterfactual': inner_increase_secondary,
        'lower risk, yield above counterfactual': inner_decrease_primary,
        'lower risk, yield below counterfactual': inner_decrease_secondary
    }
    return var_draw_strategies[category]
