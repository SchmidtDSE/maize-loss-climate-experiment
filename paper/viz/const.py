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

YIELD_MIN_VALUE = 0
YIELD_MAX_VALUE = 3000
YIELD_INCREMENT = 250

YIELD_CHANGE_MIN_VALUE = -100
YIELD_CHANGE_MAX_VALUE = 100
YIELD_CHANGE_INCREMENT = 10

RISK_MIN_VALUE = -30
RISK_MAX_VALUE = 60
RISK_INCREMENT = 10

STD_MIN_VALUE = -30
STD_MAX_VALUE = 30
STD_INCREMENT = 5

ADAPT_MIN_VALUE = -70
ADAPT_MAX_VALUE = 90
ADAPT_INCREMENT = 10

VAR_MINS = {
    'chirps': -2,
    'rhn': -4,
    'rhx': -4,
    'svp': -0.1,
    'tmax': -5,
    'tmin': -5,
    'vpd': -0.1,
    'wbgtmax': -5
}

VAR_MAXS = {
    'chirps': 2,
    'rhn': 4,
    'rhx': 4,
    'svp': 0.5,
    'tmax': 5,
    'tmin': 5,
    'vpd': 0.5,
    'wbgtmax': 5
}

VAR_INCREMENTS = {
    'chirps': 0.5,
    'rhn': 0.5,
    'rhx': 0.5,
    'svp': 0.1,
    'tmax': 1,
    'tmin': 1,
    'vpd': 0.1,
    'wbgtmax': 1
}

MAX_COUNT = 0.1

CATEGORY_COLORS = {
    'not significant': '#c0c0c0',
    'decrease': '#d95f02',
    'increase': '#1b9e77',
    'increased risk, can adapt': '#1f78b4',
    'increased risk, cant adapt': '#a6cee3',
    'decreased risk, can adapt': '#33a02c',
    'decreased risk, cant adapt': '#b2df8a',
    'increase risk, increase variability': '#1f78b4',
    'increase risk, decrease variability': '#a6cee3',
    'decrease risk, increase variability': '#33a02c',
    'decrease risk, decrease variability': '#b2df8a'
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
    'Click select to find fields (dots below) that would be good for a pilot.',
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

HISTORY_BODY_COLOR = '#1f78b4'
HISTORY_BODY_LOSS_COLOR = '#33a02c'
HISTORY_PENDING_COLOR = '#a6cee3'
