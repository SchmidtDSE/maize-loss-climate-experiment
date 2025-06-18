#!/bin/bash

# Dev container validation script
# This script validates that the dev container environment is properly set up

echo "🔍 Validating development environment..."

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python --version 2>&1)
if [[ $python_version == *"3.11"* ]]; then
    echo "✅ Python 3.11 is installed: $python_version"
else
    echo "❌ Python 3.11 not found: $python_version"
    exit 1
fi

# Check essential system tools
echo "📋 Checking system tools..."
tools=("wget" "zip" "pandoc" "git")
for tool in "${tools[@]}"; do
    if command -v "$tool" &> /dev/null; then
        echo "✅ $tool is available"
    else
        echo "❌ $tool is missing"
        exit 1
    fi
done

# Check Python packages
echo "📋 Checking Python packages..."
packages=("numpy" "pandas" "matplotlib" "luigi" "bokeh" "jinja2")
for package in "${packages[@]}"; do
    if python -c "import $package" &> /dev/null; then
        echo "✅ $package is installed"
    else
        echo "❌ $package is missing"
        exit 1
    fi
done

# Check directory structure
echo "📋 Checking directory structure..."
dirs=("paper/viz/font" "paper/outputs" "paper/viz/data" "dev-data")
for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "✅ Directory $dir exists"
    else
        echo "❌ Directory $dir is missing"
        exit 1
    fi
done

# Check font file
if [ -f "paper/viz/font/PublicSans-Regular.otf" ]; then
    echo "✅ Public Sans font is installed"
else
    echo "❌ Public Sans font is missing"
    exit 1
fi

# Check sample data (if available)
if [ -f "paper/viz/data/README.md" ] || [ -f "paper/outputs/README.md" ]; then
    echo "✅ Sample data directories are set up"
else
    echo "⚠️ Sample data may not be available (this is normal if downloads failed)"
fi

echo ""
echo "🎉 Development environment validation complete!"
echo ""
echo "You can now:"
echo "  - Develop the pipeline: bash run.sh (requires source data in dev-data/)"
echo "  - Build the paper: cd paper && bash render.sh"
echo "  - Run visualizations: cd paper/viz && python hist_viz.py"
