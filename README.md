# Maize Loss Climate Experiment
Study looking at insurance changing loss rates in climate change.

<br>

## Purpose 
This repository contains three components for a study looking at how crop insurance claims rates may change in the future within the US Corn Belt using [SCYM]() and [CHC-CMIP6]().

 - **Pipeline**: Contained within the root of this repository, this [Luigi]()-based pipeline trains neural networks and runs Monte Carlo simulations to project future insurance claims under various different parameters, outputting data to a workspace directory.
 - **Tools**: Within the `paper/viz` subdirectory, the source code for an [explorable explanation]() built using [Sketchingpy]() both creates the static visualizations for the paper and offers web-based interactive tools released to []() which allow users to iteratively engage with these results.
 - **Paper**: Within the `paper` subdirectory, a manuscript is built from the output data from the pipeline. This describes these experiments in detail with visualizations.

These are described in detail below.

<br>

## Usage
The easiest way to engage with these results is through the web-based interactive explorable explanation which is housed for the public at . The paper preprint can also be found at. We also publish our [raw pipeline output]().

<br>

## Local setup
For those wishing to extend this work, you can execute this pipeline locally by checking out this repository (). Due to the large datasets involved and various licensing constraints, there are different steps for each part of this repository.

### Local pipeline setup
The [SCYM]() and [CHC-CMIP6]() datasets are quite large and require data access permission due to licensing and privacy constraints. Therefore, by default, this work assumes those data are hosted remotely in an AWS bucket defined by the `PIPELINE_BUCKET_NAME` environment variable and we use [Coiled]() to execute the computation. First, install the Python requirements (), optionally within a [virtual environment](). Then, after setting the environment variable () and setting up Coiled, simply execute the [Luigi]() pipeline via `bash run.sh` or similar. This is the only part of the pipeline which cannot execute using fully open source components. Even so, this can also be executed independently of the other components of this repository.

### Tools
Written in [Sketchingpy](), the tools can be executed locally on your computer, in a static context for building the paper, or through a web browser. There are two options for executing the tools:

 - **Docker**: You can run the web-based visualizations through a simple Docker file in the `paper/viz` directory.
 - **Local**: You can execute the visualizations manually by running them directly as Python scripts. The entry points are `hist_viz.py`, `history_viz.py`, `results_viz_entry.py`, and `sweep_viz.py`. Simply run them without any command line arguments for defaults.

Note that the visualizations are also invoked through `paper/viz/render_images.sh` for the paper.