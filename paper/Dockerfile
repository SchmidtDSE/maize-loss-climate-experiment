FROM ubuntu:noble-20240605

RUN apt-get update
RUN apt-get install -y python3-full
RUN apt-get install -y python3-pip
RUN apt-get install -y python-is-python3
RUN apt-get install -y zip
RUN apt-get install -y wget
RUN apt-get install -y texlive-extra-utils texlive-fonts-recommended texlive-latex-base texlive-latex-extra texlive-bibtex-extra biber

RUN mkdir /workspace
WORKDIR /workspace

COPY . .

RUN pip install --break-system-packages -r requirements.txt
RUN cd viz; bash script/setup_font.sh
RUN cd img; wget https://ag-adaptation-study.pub/archive/pipeline.png

RUN wget https://github.com/jgm/pandoc/releases/download/3.1.9/pandoc-3.1.9-1-amd64.deb
RUN dpkg -i pandoc-3.1.9-1-amd64.deb

RUN apt install -y git-all

RUN python -m pip install --force-reinstall --break-system-packages git+https://github.com/tomduck/pandoc-xnos@284474574f51888be75603e7d1df667a0890504d#egg=pandoc-xnos
