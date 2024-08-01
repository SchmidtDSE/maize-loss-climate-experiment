import sys

import matplotlib.font_manager
import matplotlib.pyplot
import numpy
import pandas

NUM_ARGS = 2
USAGE_STR = 'python std_hist.py [source csv] [destination graphic]'


def main():
    if len(sys.argv) != NUM_ARGS + 1:
        print(USAGE_STR)
        sys.exit(1)

    source = sys.argv[1]
    destination = sys.argv[2]

    source = pandas.read_csv(source)
    source['percentEquivalent'] = source['projectedYieldStd'] * 100

    bins = numpy.linspace(10, 20, 20)
    font_properties = matplotlib.font_manager.FontProperties(fname='./font/PublicSans-Regular.otf')
    matplotlib.pyplot.hist(
        x=source[source['condition'] == 'historic']['percentEquivalent'],
        bins=bins,
        color='#a6cee3',
        alpha=0.5,
        label='Current'
    )
    matplotlib.pyplot.hist(
        x=source[source['condition'] == '2030_SSP245']['percentEquivalent'],
        bins=bins,
        color='#b2df8a',
        alpha=0.5,
        label='2030 Series'
    )
    matplotlib.pyplot.hist(
        x=source[source['condition'] == '2050_SSP245']['percentEquivalent'],
        bins=bins,
        color='#33a02c',
        alpha=0.5,
        label='2050 Series'
    )
    matplotlib.pyplot.xlabel('Percent From Expected Yield Equivalent to 1 Standard Deviation', fontproperties=font_properties)
    matplotlib.pyplot.ylabel('Number of Neighborhoods (4 Char Geohashes)', fontproperties=font_properties)

    matplotlib.pyplot.legend(prop=font_properties)

    ax = matplotlib.pyplot.gca()
    ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)

    for label in ax.get_xticklabels():
        label.set_fontproperties(font_properties)

    for label in ax.get_yticklabels():
        label.set_fontproperties(font_properties)

    matplotlib.pyplot.savefig(destination)


if __name__ == '__main__':
    main()
