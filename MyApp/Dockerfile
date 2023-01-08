#FROM continuumio/anaconda3:latest
FROM ubuntu:18.04

WORKDIR /home

COPY MyApp .

RUN apt-get update

RUN apt-get install -y python3.8

RUN alias python3=python3.8
RUN ln -s /usr/bin/python3.8 /usr/bin/python3
RUN pip3 install -r requirements.txt

CMD ["python", "training_model_flask.py"]



