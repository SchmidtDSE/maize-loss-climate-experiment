FROM ubuntu:noble-20240605

RUN apt-get update
RUN apt-get install -y python3-full
RUN apt-get install -y python3-pip
RUN apt-get install -y python-is-python3

RUN apt-get install -y zip
RUN apt-get install -y wget

RUN apt-get install -y nginx

RUN mkdir /service
WORKDIR /service

COPY . .

RUN bash script/setup_font.sh
RUN wget https://ag-adaptation-study.pub/archive/third_party.zip
RUN unzip -o third_party.zip

RUN bash script/prepare_deploy.sh
RUN cp -r data deploy/data
RUN cp -r font deploy/font
RUN cp -r img deploy/img
RUN cp -r third_party deploy/third_party

COPY nginx.conf /etc/nginx/nginx.conf
