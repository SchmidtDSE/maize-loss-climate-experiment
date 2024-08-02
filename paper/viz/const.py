BACKGROUND_COLOR = '#F5F5F5'
PANEL_BG_COLOR = '#FFFFFF'
BUTTON_BG_COLOR = '#E0E0E0'
HOVER_AREA_COLOR = '#F0F0F0'

INACTIVE_TEXT_COLOR = '#555555'
ACTIVE_TEXT_COLOR = '#000000'
EMBEDDED_BAR_COLOR = '#C0C0C0C0'
EMBEDDED_BAR_COLOR_TEXT = '#A0A0A0'
REFERENCE_COLOR = '#E0E0E0'

INACTIVE_BORDER = '#E0E0E0'
ACTIVE_BORDER = '#909090'
HOVER_BORDER = '#333333'

SELECT_COLOR = '#000000'
SELECTED_COLOR = '#333333'

WIDTH = 950
HEIGHT = 700
MAIN_WIDTH = 650

BUTTON_HEIGHT = 25
BUTTON_WIDTH = 120
BUTTON_WIDTH_COMPACT = 78
BUTTON_WIDTH_NARROW = 60

YIELD_MIN_VALUE = -0.3
YIELD_MAX_VALUE = 0.3
YIELD_INCREMENT = 0.05

YIELD_CHANGE_MIN_VALUE = -0.4
YIELD_CHANGE_MAX_VALUE = 0.4
YIELD_CHANGE_INCREMENT = 0.05

RISK_MIN_VALUE = -20
RISK_MAX_VALUE = 30
RISK_INCREMENT = 10

STD_MIN_VALUE = -30
STD_MAX_VALUE = 30
STD_INCREMENT = 5

ADAPT_MIN_VALUE = -70
ADAPT_MAX_VALUE = 90
ADAPT_INCREMENT = 10

VAR_MINS = {
    'chirps': -2.5,
    'rhn': -2.5,
    'rhx': -2.5,
    'svp': -2.5,
    'tmax': -2.5,
    'tmin': -2.5,
    'vpd': -2.5,
    'wbgtmax': -2.5
}

VAR_MAXS = {
    'chirps': 3.5,
    'rhn': 3.5,
    'rhx': 3.5,
    'svp': 3.5,
    'tmax': 3.5,
    'tmin': 3.5,
    'vpd': 3.5,
    'wbgtmax': 3.5
}

VAR_INCREMENTS = {
    'chirps': 0.5,
    'rhn': 0.5,
    'rhx': 0.5,
    'svp': 0.5,
    'tmax': 0.5,
    'tmin': 0.5,
    'vpd': 0.5,
    'wbgtmax': 0.5
}

MAX_COUNT = 0.1

CATEGORY_COLORS = {
    'no significant change': '#c0c0c0',
    'lower than counterfactual': '#a6cee3',
    'higher than counterfactual': '#b2df8a',
    'higher risk, can adapt': '#1f78b4',
    'higher risk, cant adapt': '#a6cee3',
    'lower risk, can adapt': '#33a02c',
    'lower risk, cant adapt': '#b2df8a',
    'higher risk, yield above counterfactual': '#a6cee3',
    'higher risk, yield below counterfactual': '#1f78b4',
    'lower risk, yield above counterfactual': '#b2df8a',
    'lower risk, yield below counterfactual': '#33a02c'
}

MONTH_NUMS = {
    'jan': 1,
    'feb': 2,
    'mar': 3,
    'apr': 4,
    'may': 5,
    'jun': 6,
    'jul': 7,
    'aug': 8,
    'sep': 9,
    'oct': 10,
    'nov': 11,
    'dec': 12
}

MAP_SCALE_NEGATIVE = ['#fde0efa0', '#f1b6daa0', '#de77aea0', '#c51b7da0']
MAP_SCALE_POSITIVE = ['#e6f5d0a0', '#b8e186a0', '#7fbc41a0', '#4d9221a0']

SELECTION_INSTRUCTION = ' '.join([
    'Click select to find neighborhoods (dots below) good for a pilot.',
    'Toggle yield, risk and adaptation or map vs scatter.',
    'Do different views change your answer?'
])

FONT_SRC = './font/PublicSans-Regular.otf'

HISTORY_INSTRUCTION = ' '.join([
    'Click to change a yield for a single year.',
    'How do claims change with stdev or average?',
    'Low or high stability?',
    'What behavior does std or average encourage?'
])

RATES_INSTRUCTION = ' '.join([
    'Modify the variables below to see what leads to the highest subsidy.',
    'How does increasing or decreasing a grower APH change that outlook?'
])

HISTORY_BODY_COLOR = '#1f78b4'
HISTORY_BODY_LOSS_COLOR = '#33a02c'
HISTORY_PENDING_COLOR = '#a6cee3'
