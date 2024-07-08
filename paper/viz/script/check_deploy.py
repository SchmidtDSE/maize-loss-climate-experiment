import os
import os.path
import sys

USAGE_STR = 'python check_deploy.py [directory]'
NUM_ARGS = 1


def main():
    if len(sys.argv) != NUM_ARGS + 1:
        print(USAGE_STR)
        sys.exit(1)

    directory = sys.argv[1]

    with open(os.path.join(directory, 'index.html')) as f:
        index_contents = f.read()

    assert 'results_viz.pyscript' in index_contents

    with open(os.path.join(directory, 'results_viz.pyscript')) as f:
        script_contents = f.read()

    assert 'ResultsVizPresenter' in script_contents

    print('Checked deployment.')


if __name__ == '__main__':
    main()
