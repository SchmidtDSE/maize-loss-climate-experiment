# Development Container

This directory contains the development container configuration for the Maize Loss Climate Experiment project.

## What it provides

The dev container automatically sets up a complete development environment with:

- **Python 3.11** with all project dependencies installed
- **LaTeX and Pandoc** for paper building
- **System dependencies** (zip, wget, build tools)
- **Development tools** (linting, testing, code formatting)
- **Sample data** downloaded automatically for visualization development
- **Public Sans font** configured for visualizations

## Usage
The dev contisner can be used across many environments and platforms but we provide some commons ones here.

### GitHub Codespaces
1. Click the "Code" button on GitHub
2. Select "Open with Codespaces"
3. Wait for the environment to build and configure

### VS Code with Dev Containers Extension
1. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
2. Open the project in VS Code
3. Click "Reopen in Container" when prompted
4. Or use Command Palette: "Dev Containers: Reopen in Container"

### Docker Compose
```bash
cd .devcontainer
docker-compose up -d
docker-compose exec dev bash
```

### Manual Docker Build
```bash
docker build -t maize-experiment .devcontainer
docker run -it -v $(pwd):/workspaces/maize-loss-climate-experiment maize-experiment
```

## Environment

The container sets up these environment variables:
- `USE_AWS=0` - Configured for local development
- `SOURCE_DATA_LOC=/workspaces/maize-loss-climate-experiment/dev-data` - Directory for source data

## Development workflow

Once the container is running:

1. **Pipeline development**: Place your SCYM and CHC-CMIP6 data in `dev-data/` and run `bash run.sh`
2. **Paper development**: Run `cd paper && bash render.sh`
3. **Visualization development**: Run `cd paper/viz && python hist_viz.py` (or other viz scripts)

## Ports

The following ports are forwarded for development:
- `8000` - General web development
- `8080` - Alternative web port
- `8888` - Jupyter notebooks (if used)

## Extensions

The container includes VS Code extensions for:
- Python development and linting
- Jupyter notebook support
- YAML and JSON editing
- Makefile support
