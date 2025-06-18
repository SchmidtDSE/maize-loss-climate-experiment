#!/bin/bash

# Post-create script for setting up the development environment
set -e

echo "Setting up development environment..."

# Create font build directory and download font
mkdir -p font_build
cd font_build
echo "Downloading Public Sans font..."
wget -q https://github.com/uswds/public-sans/releases/download/v2.001/public-sans-v2.001.zip
unzip -o -q public-sans-v2.001.zip

# Set up font in viz directory
mkdir -p ../paper/viz/font
cp fonts/otf/PublicSans-Regular.otf ../paper/viz/font/PublicSans-Regular.otf

# Clean up
cd ..
rm -rf font_build

echo "Downloading sample data for development..."

# Download and set up sample data for development (using the public versions)
# This provides a basic setup for development without requiring full dataset access

# Download pipeline summary outputs
echo "Downloading pipeline summary outputs..."
wget -q https://ag-adaptation-study.pub/archive/outputs.zip || echo "Warning: Could not download outputs.zip - continuing without sample data"
if [ -f outputs.zip ]; then
    unzip -o -q outputs.zip
    mv outputs paper/outputs
    rm outputs.zip
    echo "✓ Pipeline summary outputs downloaded"
fi

# Download pipeline detailed outputs  
echo "Downloading pipeline detailed outputs..."
wget -q https://ag-adaptation-study.pub/archive/data.zip || echo "Warning: Could not download data.zip - continuing without sample data"
if [ -f data.zip ]; then
    unzip -o -q data.zip
    mv data paper/viz/data
    rm data.zip
    echo "✓ Pipeline detailed outputs downloaded"
fi

# Download additional sweep data
if [ -d paper/viz/data ]; then
    cd paper/viz/data
    echo "Downloading sweep data..."
    wget -q http://ag-adaptation-study.pub/data/sweep_ag_all.csv || echo "Warning: Could not download sweep data"
    cd ../../../
    echo "✓ Sweep data downloaded"
fi

# Download third party dependencies for visualization
echo "Downloading third party dependencies..."
cd paper/viz
wget -q https://ag-adaptation-study.pub/archive/third_party.zip || echo "Warning: Could not download third_party.zip - continuing without third party dependencies"
if [ -f third_party.zip ]; then
    unzip -o -q third_party.zip
    rm third_party.zip
    echo "✓ Third party dependencies downloaded"
fi
cd ../..

# Create a placeholder for source data directory
mkdir -p dev-data
echo "# Development Data Directory" > dev-data/README.md
echo "This directory is for development data. Place your SCYM and CHC-CMIP6 geotiff files here for local development." >> dev-data/README.md

# Verify installations
echo "Verifying installations..."
python --version
pip --version
pandoc --version

echo "Development environment setup complete!"
echo ""
echo "Available components:"
echo "  - Pipeline: Run 'bash run.sh' (requires source data in dev-data/)"
echo "  - Paper: Run 'cd paper && bash render.sh'"
echo "  - Viz: Run 'cd paper/viz && python hist_viz.py' (or other viz scripts)"
echo ""
echo "For pipeline development, place your source data in the dev-data/ directory"
echo "and set SOURCE_DATA_LOC environment variable if needed."
echo ""
echo "Run '.devcontainer/validate-setup.sh' to validate the environment setup."
