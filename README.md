# Maize Loss Climate Experiment
Study looking at how climate change may impact loss rates for insurance by simulating maize outcomes via a neural network and Monte Carlo. This includes interactive tools to understand results and a pipeline to build a paper discussing findings.

<br>

## Purpose
This repository contains three components for a study looking at how crop insurance claims rates may change in the future within the US Corn Belt using [SCYM](https://www.sciencedirect.com/science/article/pii/S0034425715001637) and [CHC-CMIP6](https://www.chc.ucsb.edu/data/chc-cmip6).

 - **Pipeline**: Contained within the root of this repository, this [Luigi](https://luigi.readthedocs.io/en/stable/)-based pipeline trains neural networks and runs Monte Carlo simulations to project future insurance claims under various different parameters, outputting data to a workspace directory.
 - **Tools**: Within the `paper/viz` subdirectory, the source code for an [explorable explanation](https://worrydream.com/ExplorableExplanations/) built using [Sketchingpy](https://sketchingpy.org/) both creates the static visualizations for the paper and offers web-based interactive tools released to [ag-adaptation-study.pub](https://ag-adaptation-study.pub/) which allow users to iteratively engage with these results.
 - **Paper**: Within the `paper` subdirectory, a manuscript is built from the output data from the pipeline. This describes these experiments in detail with visualizations.

These are described in detail below.

<br>

## Usage
The easiest way to engage with these results is through the web-based interactive explorable explanation which is housed for the public at [ag-adaptation-study.pub](https://ag-adaptation-study.pub/). The paper preprint can also be found at https://arxiv.org/abs/2408.02217. We also publish our [raw pipeline output](https://ag-adaptation-study.pub/archive/output.zip). Otherwise, see local setup.

<br>

## Local setup
For those wishing to extend this work, you can execute this pipeline locally by checking out this repository (`git clone git@github.com:SchmidtDSE/maize-loss-climate-experiment.git`).

### Dev Container (Recommended)
The easiest way to get started with development is using the provided dev container, which automatically sets up all dependencies for pipeline, paper, and visualization development:

1. **GitHub Codespaces**: Click the "Code" button and select "Open with Codespaces" for instant cloud-based development.
2. **VS Code with Dev Containers**: Open the repository in VS Code and click "Reopen in Container" when prompted (requires Docker and the Dev Containers extension).
3. **Local Docker**: Clone the repository and run `docker build -t maize-experiment .devcontainer` then `docker run -it -v $(pwd):/workspaces/maize-loss-climate-experiment maize-experiment`.

The dev container includes:
- Python 3.11 with all project dependencies pre-installed
- LaTeX and Pandoc for paper building
- Sample data for visualization development
- Development tools (linting, testing)
- All system dependencies configured

After the container starts, you'll have a fully configured environment ready for development on any component.

### Local pipeline
First, get access to the [SCYM](https://www.sciencedirect.com/science/article/pii/S0034425715001637) and [CHC-CMIP6](https://www.chc.ucsb.edu/data/chc-cmip6) datasets and download all of the geotiffs to an AWS S3 Bucket or another location which can be accessed via the file system. This will allow you to choose from two execution options:

 - **Setup for AWS**: This will execute if the `USE_AWS` environment variable is set to 1. This assumes data are hosted remotely in an AWS bucket defined by the `SOURCE_DATA_LOC` environment variable and we use [Coiled](https://www.coiled.io/) to execute the computation. After setting the environment variables for access credientials (`AWS_ACCESS_KEY` and `AWS_ACCESS_SECRET`) and setting up Coiled, simply execute the [Luigi](https://luigi.readthedocs.io/en/stable/) pipeline as described below.
 - **Setup for local**: If the `USE_AWS` environment variable is set to 0, this will run using a local Dask cluster. This assumes that `SOURCE_DATA_LOC` is a path to the directory housing the input geotiffs. After setting up Coiled, simply execute the [Luigi](https://luigi.readthedocs.io/en/stable/) pipeline as described below.

You can then execute either by:

 - **Run directly**: First, install the Python requirements (`pip install -r requirements.txt`) optionally within a [virtual environment](https://python-guide-es.readthedocs.io/es/guide-es/dev/virtualenvs.html). Then, simply execute `bash run.sh` to execute the pipeline from start to finish. See also `breakpoint_tasks.py` for [Luigi](https://luigi.readthedocs.io/en/stable/) targets for running subsets of the pipeline.
 - **Run through Docker**: Simply execute `bash run_docker.sh` to execute the pipeline from start to finish. See also `breakpoint_tasks.py` for [Luigi](https://luigi.readthedocs.io/en/stable/) targets for running subsets of the pipeline and update `run.sh` which is executed within the container. Note that this will operate on the `workspace` directory.

A summary of the pipeline is created in `stats.json`. See local package below for use in other repository components such as the interactive tools or paper rendering. **Users may optionally skip some expensive steps by placing the files from https://zenodo.org/records/14533227 into the `workspace` directory.**

### Interactive tools
Written in [Sketchingpy](https://sketchingpy.org/), the tools can be executed locally on your computer, in a static context for building the paper, or through a web browser. First, one needs to get data from the pipeline or download prior results:

 - **Download prior results**: Retrieve the [latest results](https://ag-adaptation-study.pub/archive/data.zip) and move them into the viz directory (`paper/viz/data`). Simply use wget to gather model outputs when in the `paper/viz directory` as so: `wget https://ag-adaptation-study.pub/archive/data.zip; unzip data.zip`. If using prior sweep results, download full sweep information like so: `cd data; wget http://ag-adaptation-study.pub/data/sweep_ag_all.csv; cd ..`.
 - **Use your own results**: Update the output data per instructions regarding local package below.

There are two options for executing the tools:

 - **Docker**: You can run the web-based visualizations through a simple Docker file in the `paper/viz` directory (`bash run_docker.sh`).
 - **Local apps**: You can execute the visualizations manually by running them directly as Python scripts. The entry points are `hist_viz.py`, `history_viz.py`, `results_viz_entry.py`, and `sweep_viz.py`. Simply run them without any command line arguments for defaults. Note you may need to install python dependencies (`pip install -r requirements.txt`).

Note that the visualizations are also invoked through `paper/viz/render_images.sh` for the paper.

### Paper
Due to the complexities of the software install, the only officially supported way to build the paper is through the Docker image. First update the data:

 - **Download prior results**: Retrieve the [latest results](https://ag-adaptation-study.pub/archive/outputs.zip) and move them into the paper directory (`paper/outputs`).
 - **Use your own results**: Update the output data per instructions regarding local package below.

Then, execute `render_docker.sh` to drop the results into the `paper_rendered` directory.

### Local package
Instead of retrieving data from https://ag-adaptation-study.pub, you can use your own pipeline data outputs by running `bash package.sh`. This will produce the `data` and `outputs` sub-directories inside of a new `package` directory which can be used for the interactive tools and paper rendering respectively.

### Alternative: Manual setup
If you prefer not to use the dev container, you can manually set up each component following the individual setup instructions below, though this requires more configuration steps.

<br>

## Examples

The following section provides a "cookbook" of common examples for how to use these tools for the most common scenarios.

### Review existing results
If wanting to review the current outputs, simply navigate to [https://ag-adaptation-study.pub](https://ag-adaptation-study.pub). No additional software is required. 

### Run interactive tools
To use the existing outputs and run the interactive tools locally after cloning this repository, gather the data and run the visualization scripts.

```
$ cd paper/viz
$ wget https://ag-adaptation-study.pub/archive/data.zip
$ unzip data.zip
$ cd data
$ wget http://ag-adaptation-study.pub/data/sweep_ag_all.csv
$ cd ..
$ pip install -r requirements.txt
$ python hist_viz.py
```

This runs the historgram visualization but `hist_viz.py`, `history_viz.py`, `rates_viz.py`, `results_viz_entry.py`, and `sweep_viz.py` are all available.

### Execute the pipeline locally via Docker
The following will execute the entire pipeline locally after having placed [SCYM](https://www.sciencedirect.com/science/article/pii/S0034425715001637) and [CHC-CMIP6](https://www.chc.ucsb.edu/data/chc-cmip6) in a local directory (assumed to be `path/to/data` below).

```
$ export USE_AWS=0
$ export SOURCE_DATA_LOC=path/to/data
$ bash run_docker.sh
```

<br>

## Testing
As part of CI / CD and for local development, the following are required to pass for both the pipeline in the root of this repository and the interactives written in Python at `paper/viz`:

 - **pyflakes**: Run `pyflakes *.py` to check for likely non-style code isses.
 - **pycodestyle**: Run `pycodestyle *.py` to enforce project coding style guidelines.

The pipeline also offers unit tests (`nose2` in root) for the pipeline. For the visualizations, tests happen by running the interactives headless (`bash render_images.sh; bash script/check_images.sh`).

<br>

## Deployment
To deploy changes to production, CI / CD will automatically release to ag-adaptation-study.pub once merged on `main`.

<br>

## Development standards
Where possible, please follow the [Python Google Style Guide](https://google.github.io/styleguide/pyguide.html) unless an override is provided in `setup.cfg`. Docstrings and type hints are required for all top-level or public members but not currently enforced for private members. [JSDoc](https://jsdoc.app/) is required for top level members. Docstring / JSDoc not required for "utility" code.

<br>

## Data
We make publicly available both inputs and outputs to our modeling. Due to size, some of these are archived at [ag-adaptation-study.pub](https://ag-adaptation-study.pub) while others are deposited into [Zenodo](https://zenodo.org/doi/10.5281/zenodo.13356980).

### CHC-CMIP6
Our derivative dataset from CHC-CMIP6 is available at [climate.csv](https://ag-adaptation-study.pub/data/climate.csv) and our [Zenodo](https://zenodo.org/doi/10.5281/zenodo.13356980). These are aggregated and preprocessed as used within our modeling.

### USDA RMA SOB
Note that an archive of [UDSA Risk Management Agency (RMA) Summary of Business (SOB) data](https://www.rma.usda.gov/tools-reports/summary-of-business) is also provided at our [usda_rma_sob.zip](https://ag-adaptation-study.pub/data/usda_rma_sob.zip). As a relatively large supplemental dataset, this is not currently in Zenodo. In addition to original format, all SOB datasets are given in Avro format where possible with standardized formatting / encoding. A subset of these data are considered within our paper as supporting evidence. See the README within the data archive for further details.

### Yield estimations (SCYM)
Our derivative SCYM yield estimations at neighborhood-level are avialable at [Zenodo](https://zenodo.org/doi/10.5281/zenodo.13356980).

### Model outputs
The following model outputs are made available both through our website:

 - [export_claims.csv](https://ag-adaptation-study.pub/data/export_claims.csv]): Information about the claims rate under different conditions.
 - [sim_hist.csv](https://ag-adaptation-study.pub/data/sim_hist.csv): Information about simulation-wide yield distributions under different conditions.
 - [sweep_ag_all.csv](https://ag-adaptation-study.pub/data/sweep_ag_all.csv): Information about sweep outcomes and model performance.
 - [tool.csv](https://ag-adaptation-study.pub/data/tool.csv): Geographically specific information about simulation outcomes at the 4 character geohash level.

As smaller payloads, these are also included in our [Zenodo](https://zenodo.org/doi/10.5281/zenodo.13356980).

<br>

## Open source
The pipeline, interactives, and paper can be executed independently and have segregated dependencies. We thank all of our open source dependencies for their contribution.

### Pipeline dependencies
The pipeline uses the following open source dependencies:

 - [bokeh](https://bokeh.org/) under the [BSD 3-Clause License](https://github.com/bokeh/bokeh?tab=BSD-3-Clause-1-ov-file#readme).
 - [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) under the [Apache v2 License](https://github.com/boto/boto3/blob/develop/LICENSE).
 - [dask](https://www.dask.org/) under the [BSD 3-Clause License](https://github.com/dask/dask/blob/main/LICENSE.txt).
 - [fiona](https://fiona.readthedocs.io/en/stable/) under the [BSD License](https://github.com/Toblerity/Fiona/blob/master/LICENSE.txt).
 - [geolib](https://github.com/joyanujoy/geolib) under the [MIT License](https://github.com/joyanujoy/geolib/blob/master/LICENSE).
 - [geotiff](https://github.com/KipCrossing/geotiff) under the [LGPL License](https://github.com/KipCrossing/geotiff/blob/main/LICENSE).
 - [geotiff](https://github.com/KipCrossing/geotiff) under the [LGPL License](https://github.com/KipCrossing/geotiff/blob/main/LICENSE).
 - [imagecodecs](https://pypi.org/project/imagecodecs/) under the [BSD 3-Clause License](https://github.com/cgohlke/imagecodecs?tab=BSD-3-Clause-1-ov-file#readme).
 - [keras](https://keras.io/) under the [Apache v2 License](https://github.com/keras-team/keras/blob/master/LICENSE).
 - [libgeohash](https://github.com/bashhike/libgeohash) under the [MIT License](https://github.com/bashhike/libgeohash?tab=MIT-1-ov-file#readme).
 - [Luigi](https://luigi.readthedocs.io/en/stable/index.html) under the [Apache v2 License](https://github.com/spotify/luigi/blob/master/LICENSE).
 - [NumPy](https://numpy.org/) under the [BSD License](https://github.com/numpy/numpy/blob/main/LICENSE.txt).
 - [Pandas](https://pandas.pydata.org/) under the [BSD License](https://github.com/pandas-dev/pandas/blob/main/LICENSE).
 - [Pathos](https://github.com/uqfoundation/pathos) under the [BSD License](https://github.com/uqfoundation/pathos/blob/master/LICENSE).
 - [requests](https://requests.readthedocs.io/en/latest/) under the [Apache v2 License](https://github.com/psf/requests?tab=Apache-2.0-1-ov-file#readme).
 - [scipy](https://scipy.org/) under the [BSD License](https://github.com/scipy/scipy/blob/main/LICENSE.txt).
 - [shapely](https://shapely.readthedocs.io/en/stable/) by Sean Gillies, Casper van der Wel, and Shapely Contributors under the [BSD License](https://github.com/shapely/shapely/blob/main/LICENSE.txt).
 - [tensorflow](https://www.tensorflow.org/) under the [Apache v2 License](https://github.com/tensorflow/tensorflow?tab=Apache-2.0-1-ov-file#readme).
 - [toolz](https://github.com/pytoolz/toolz/) under the [BSD License](https://github.com/pytoolz/toolz/blob/master/LICENSE.txt).

Use of [Coiled](https://www.coiled.io/) is optional.

### Tools and visualizations
Both the interactives and static visualization generation use the following:

 - [Jinja](https://jinja.palletsprojects.com/en/3.1.x/) under the [BSD 3-Clause License](https://jinja.palletsprojects.com/en/3.1.x/license/).
 - [Matplotlib](https://matplotlib.org/) under the [PSF License](https://matplotlib.org/stable/users/project/license.html).
 - [NumPy](https://numpy.org/) under the [BSD License](https://github.com/numpy/numpy/blob/main/LICENSE.txt).
 - [Pandas](https://pandas.pydata.org/) under the [BSD License](https://github.com/pandas-dev/pandas/blob/main/LICENSE).
 - [Pillow](https://python-pillow.org/) under the [HPND License](https://github.com/python-pillow/Pillow?tab=License-1-ov-file#readme).
 - [pygame-ce](https://pyga.me/) under the [LGPL License](https://pyga.me/docs/LGPL.txt).
 - [Sketchingpy](https://sketchingpy.org/) under the [BSD 3-Clause License](https://codeberg.org/sketchingpy/Sketchingpy/src/branch/main/LICENSE.md).
 - [toolz](https://github.com/pytoolz/toolz/) under the [BSD License](https://github.com/pytoolz/toolz/blob/master/LICENSE.txt).

The web version also uses:

 - es.js under the [ISC License (Andrea Giammarchi)](https://en.wikipedia.org/wiki/ISC_license).
 - [micropip](https://github.com/pyodide/micropip) under the [MPL 2.0 License](https://github.com/pyodide/micropip/blob/main/LICENSE).
 - [packaging](https://packaging.pypa.io/en/stable/) under the [BSD License](https://github.com/pypa/packaging/blob/main/LICENSE.BSD).
 - [Popper](https://popper.js.org/docs/v2/) under the [MIT License](https://github.com/floating-ui/floating-ui?tab=MIT-1-ov-file#readme).
 - [Pyodide](https://github.com/pyodide/pyodide) under the [MPL 2.0 License](https://github.com/pyodide/pyodide/blob/main/LICENSE).
 - [Pyscript](https://pyscript.net/) under the [Apache v2 License](https://pyscript.github.io/docs/2023.12.1/license/).
 - [Sketchingpy](https://sketchingpy.org/) under the [BSD 3-Clause License](https://codeberg.org/sketchingpy/Sketchingpy/src/branch/main/LICENSE.md).
 - [Tabby](https://github.com/cferdinandi/tabby) under the [MIT License](https://github.com/cferdinandi/tabby?tab=MIT-1-ov-file#readme).
 - [Tippy.js](https://atomiks.github.io/tippyjs/) under the [MIT License](https://github.com/atomiks/tippyjs?tab=MIT-1-ov-file#readme).
 - [toml (Jak Wings)](https://www.npmjs.com/package/tomlify-j0.4?activeTab=readme) under the [MIT License](https://www.npmjs.com/package/tomlify-j0.4?activeTab=code).
 - [ua-parser 1.0.36](https://uaparser.js.org/) under the [MIT License](https://www.npmjs.com/package/ua-parser-js).

### Paper
The paper uses the following open source dependencies to build the manuscript:

 - [Jinja](https://jinja.palletsprojects.com/en/3.1.x/) under the [BSD 3-Clause License](https://jinja.palletsprojects.com/en/3.1.x/license/).
 - [Matplotlib](https://matplotlib.org/) under the [PSF License](https://matplotlib.org/stable/users/project/license.html).
 - [NumPy](https://numpy.org/) under the [BSD License](https://github.com/numpy/numpy/blob/main/LICENSE.txt).
 - [Pandas](https://pandas.pydata.org/) under the [BSD License](https://github.com/pandas-dev/pandas/blob/main/LICENSE).
 - [Pillow](https://python-pillow.org/) under the [HPND License](https://github.com/python-pillow/Pillow?tab=License-1-ov-file#readme).
 - [Sketchingpy](https://sketchingpy.org/) under the [BSD License](https://codeberg.org/sketchingpy/Sketchingpy/src/branch/main/LICENSE.md) including the packages included in its stand alone hosting archive.
 - [toolz](https://github.com/pytoolz/toolz/) under the [BSD License](https://github.com/pytoolz/toolz/blob/master/LICENSE.txt).

Users may optionally leverage [Pandoc](https://pandoc.org/) as an executable (not linked) under the [GPL](https://www.gnu.org/licenses/gpl-3.0.html) but any tool converting markdown to other formats is acceptable or the paper can be built as Markdown only without Pandoc. That said, for those using Pandoc, scripts may also use [pandoc-fignos](https://github.com/tomduck/pandoc-fignos) under the [GPL License](https://github.com/tomduck/pandoc-fignos?tab=GPL-3.0-1-ov-file#readme) and [pandoc-tablenos](https://github.com/tomduck/pandoc-tablenos) under the [GPL License](https://github.com/tomduck/pandoc-tablenos?tab=GPL-3.0-1-ov-file#readme).

### Other runtime dependencies
Some executions may also use:

 - [Docker](https://docs.docker.com/engine/) under the [Apache v2 License](https://github.com/moby/moby/blob/master/LICENSE).
 - [Docker Compose](https://docs.docker.com/compose/) under the [Apache v2 License](https://github.com/docker/compose/blob/main/LICENSE).
 - [Nginx](https://nginx.org/en/) under a [BSD-like License](https://nginx.org/LICENSE).
 - [OpenRefine](https://openrefine.org/) under the [BSD License](https://github.com/OpenRefine/OpenRefine/blob/master/LICENSE.txt).

### Other sources
We also use:

 - [Color Brewer](https://colorbrewer2.org/#type=sequential&scheme=BuGn&n=3) under the [Apache v2 License](https://github.com/axismaps/colorbrewer/).
 - [Public Sans](https://public-sans.digital.gov/) under the [CC0 License](https://github.com/uswds/public-sans/blob/develop/LICENSE.md).

GitHub Copilot used for some post-publication steps, largely to prepare materials for presentations.

<br>

## License
Code is released under BSD 3-Clause and data under CC-BY-NC 4.0 International. Please see `LICENSE.md`.
