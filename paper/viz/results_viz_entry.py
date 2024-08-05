import sys

import config_buttons
import results_viz

ARG_NAMES = [
    'data',
    'scenario',
    'metric',
    'viz',
    'p',
    'adj',
    'filter',
    'var',
    'month',
    'loss',
    'climate',
    'output'
]
NUM_ARGS = len(ARG_NAMES) + 1
USAGE_STR = 'python results_viz_entry.py ' + ' '.join(map(lambda x: '[%s]' % x, ARG_NAMES))


def main():
    if len(sys.argv) == 1:
        presenter = results_viz.ResultsVizPresenter('Ag Adaptation Experiment', None)
    elif len(sys.argv) != NUM_ARGS:
        print(USAGE_STR)
        sys.exit(1)
    else:
        data_loc = sys.argv[1]
        scenario = sys.argv[2] + ' series'
        metric = sys.argv[3]
        visualization = sys.argv[4]
        threshold = 'p <  %s' % sys.argv[5]
        adjustment = sys.argv[6]
        sig_filter = 'significant only' if sys.argv[7] == 'significant' else 'all'
        var = 'no var' if sys.argv[8] == 'none' else sys.argv[8]
        month = sys.argv[9]
        loss = sys.argv[10] + '% cov'
        climate_loc = sys.argv[11]
        output_loc = sys.argv[12]

        default_configuration = config_buttons.Configuration(
            scenario,
            'Avg All Years',
            metric,
            visualization,
            threshold,
            adjustment,
            sig_filter,
            var,
            month,
            loss
        )

        presenter = results_viz.ResultsVizPresenter(
            'Ag Adaptation Experiment',
            None,
            default_configuration=default_configuration,
            data_loc=data_loc,
            climate_loc=climate_loc,
            output_loc=output_loc
        )

        assert presenter is not None


if __name__ == '__main__':
    main()
