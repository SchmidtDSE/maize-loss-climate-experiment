FROM ubuntu:noble-20240605

RUN apt-get update
RUN apt-get install -y python3-full
RUN apt-get install -y python3-pip
RUN apt-get install -y python-is-python3

RUN mkdir /pipeline
WORKDIR /pipeline

COPY *.py .
COPY *.sh .
COPY *.txt .

RUN pip install --user --break-system-packages -r requirements.txt
