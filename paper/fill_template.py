import json
import sys

import jinja2

USAGE_STR = 'python fill_template.py [source] [template vals] [destination]'
NUM_ARGS = 3


def main():
    if len(sys.argv) != NUM_ARGS + 1:
        print(USAGE_STR)
        sys.exit(1)

    source_loc = sys.argv[1]
    template_loc = sys.argv[2]
    destination_loc = sys.argv[3]

    with open(source_loc) as f:
        source_contents = f.read()

    loader = jinja2.BaseLoader()
    template = jinja2.Environment(
        loader=loader,
        comment_start_string='{//',
        comment_end_string='//}'
    ).from_string(source_contents)

    with open(template_loc) as f:
        template_vals = json.load(f)

    rendered = template.render(**template_vals)

    with open(destination_loc, 'w') as f:
        f.write(rendered)


if __name__ == '__main__':
    main()
