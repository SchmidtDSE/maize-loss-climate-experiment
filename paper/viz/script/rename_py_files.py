import os
import os.path
import sys

USAGE_STR = 'python rename_py_files.py [directory]'
NUM_ARGS = 1


def main():
    if len(sys.argv) != NUM_ARGS + 1:
        print(USAGE_STR)
        sys.exit(1)

    root = sys.argv[1]
    all_items = os.listdir(root)
    py_items = filter(lambda x: x.endswith('.py'), all_items)
    fill_py_items = map(lambda x: os.path.join(root, x), py_items)
    tasks = map(lambda x: {
        'prior': x,
        'new': x[:-3] + '.pyscript'
    }, fill_py_items)

    for task in tasks:
        os.rename(task['prior'], task['new'])


if __name__ == '__main__':
    main()
