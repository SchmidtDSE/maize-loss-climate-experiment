# Maize Loss Climate Experiment
Study looking at how climate change may change loss rates for insurance by simulating maize outcomes via a neural network and Monte Carlo.

<br>

## Purpose 
This repository contains three components for a study looking at how crop insurance claims rates may change in the future within the US Corn Belt using [SCYM]() and [CHC-CMIP6]().

 - **Pipeline**: Contained within the root of this repository, this [Luigi]()-based pipeline trains neural networks and runs Monte Carlo simulations to project future insurance claims under various different parameters, outputting data to a workspace directory.
 - **Tools**: Within the `paper/viz` subdirectory, the source code for an [explorable explanation]() built using [Sketchingpy]() both creates the static visualizations for the paper and offers web-based interactive tools released to []() which allow users to iteratively engage with these results.
 - **Paper**: Within the `paper` subdirectory, a manuscript is built from the output data from the pipeline. This describes these experiments in detail with visualizations.

These are described in detail below.

<br>

## Usage
The easiest way to engage with these results is through the web-based interactive explorable explanation which is housed for the public at . The paper preprint can also be found at. We also publish our [raw pipeline output](). Otherwise, see local setup.

<br>

## Local setup
For those wishing to extend this work, you can execute this pipeline locally by checking out this repository ().

### Local pipeline
First, get access to the [SCYM]() and [CHC-CMIP6]() datasets and download all of the geotiffs to an AWS S3 Bucket or another location which can be accessed via the file system. This will allow you to choose from two execution options:

 - **Setup for Coiled**: This will execute via AWS if the `USE_AWS` environment variable is set to 1. This assumes data are hosted remotely in an AWS bucket defined by the `SOURCE_DATA_LOC` environment variable and we use [Coiled]() to execute the computation. After setting the environment variables for access credientials (`AWS_ACCESS_KEY` and `AWS_ACCESS_SECRET`) and setting up Coiled, simply execute the [Luigi]() pipeline as described below.
 - **Setup for local**: If the `USE_AWS` environment variable is set to 0, this will run using a local Dask cluster. This assumes that `SOURCE_DATA_LOC` is a path to the directory housing the input geotiffs. After setting up Coiled, simply execute the [Luigi]() pipeline as described below.

You can then execute either by:

 - **Run directly**: First, install the Python requirements () optionally within a [virtual environment](). Then, simply execute `bash run.sh` to execute the pipeline from start to finish. See also `breakpoint_tasks.py` for [Luigi]() targets for running subsets of the pipeline. 
 - **Run through Docker**: Simply execute `bash run_docker.sh` to execute the pipeline from start to finish. See also `breakpoint_tasks.py` for [Luigi]() targets for running subsets of the pipeline and update `run.sh` which is executed within the container. Note that this will operate on the `workspace` directory.

A summary of the pipeline is created in `stats.json`.

### Interactive tools
Written in [Sketchingpy](), the tools can be executed locally on your computer, in a static context for building the paper, or through a web browser. First, one needs to get data from the pipeline or download prior results:

 - **Download prior results**: retrieve the 

There are two options for executing the tools:

 - **Docker**: You can run the web-based visualizations through a simple Docker file in the `paper/viz` directory ().
 - **Local**: You can execute the visualizations manually by running them directly as Python scripts. The entry points are `hist_viz.py`, `history_viz.py`, `results_viz_entry.py`, and `sweep_viz.py`. Simply run them without any command line arguments for defaults.

Note that the visualizations are also invoked through `paper/viz/render_images.sh` for the paper.