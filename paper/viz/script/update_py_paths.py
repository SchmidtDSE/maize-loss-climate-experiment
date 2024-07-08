import os
import sys

USAGE_STR = 'python update_py_paths.py [input] [output]'
NUM_ARGS = 2


def main():
    if len(sys.argv) != NUM_ARGS + 1:
        print(USAGE_STR)
        sys.exit(1)

    input_loc = sys.argv[1]
    output_loc = sys.argv[2]

    with open(input_loc) as f:
        original_contents = f.read()

    new_contents = original_contents.replace(
        '.py?v=EPOCH":',
        '.pyscript?v=EPOCH":'
    )

    with open(output_loc, 'w') as f:
        f.write(new_contents)


if __name__ == '__main__':
    main()
