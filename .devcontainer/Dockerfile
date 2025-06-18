FROM mcr.microsoft.com/devcontainers/python:3.11

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        zip \
        wget \
        curl \
        build-essential \
        git \
        # LaTeX and Pandoc dependencies for paper building
        texlive-extra-utils \
        texlive-fonts-recommended \
        texlive-latex-base \
        texlive-latex-extra \
        texlive-bibtex-extra \
        biber \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install Pandoc
RUN wget https://github.com/jgm/pandoc/releases/download/3.1.9/pandoc-3.1.9-1-amd64.deb \
    && dpkg -i pandoc-3.1.9-1-amd64.deb \
    && rm pandoc-3.1.9-1-amd64.deb

# Install Python development tools
RUN pip install --upgrade pip \
    && pip install pycodestyle pyflakes nose2

# Set up workspace
WORKDIR /workspaces/maize-loss-climate-experiment

# Copy requirements files first for better Docker layer caching
COPY requirements.txt .
COPY paper/requirements.txt ./paper/
COPY paper/viz/requirements.txt ./paper/viz/

# Install all Python dependencies
RUN pip install -r requirements.txt \
    && pip install -r paper/requirements.txt \
    && pip install -r paper/viz/requirements.txt

# Install additional Pandoc extensions
RUN pip install --force-reinstall git+https://github.com/tomduck/pandoc-xnos@284474574f51888be75603e7d1df667a0890504d#egg=pandoc-xnos \
    && pip install pandoc-eqnos

# Create directories for development data
RUN mkdir -p paper/outputs paper/viz/data paper/viz/font paper/img

# Switch to vscode user for the remaining operations
USER vscode

# Set up environment variables for development
ENV USE_AWS=0
ENV SOURCE_DATA_LOC=/workspaces/maize-loss-climate-experiment/dev-data
