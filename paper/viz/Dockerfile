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

ENV BEZIER_IGNORE_VERSION_CHECK=true
RUN pip install --user --break-system-packages -r requirements.txt
RUN bash script/setup_font.sh
RUN wget https://ag-adaptation-study.pub/archive/third_party.zip
RUN unzip third_party.zip

RUN bash script prepare_deploy
RUN cp data deploy/data
RUN cp font deploy/font
RUN cp img deploy/img
RUN cp third_party deploy/third_party

COPY nginx.conf /etc/nginx/nginx.conf
