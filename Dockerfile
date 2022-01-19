FROM tensorflow/tensorflow:latest

COPY ./requirements.txt /

RUN pip3 install -r requirements.txt 

COPY ./ /

RUN python main.py

WORKDIR /