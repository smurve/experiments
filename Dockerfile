FROM tensorflow/tensorflow:latest-gpu
MAINTAINER Wolfgang Giersche
RUN apt-get update
RUN pip install --upgrade pip
RUN pip install pymongo flask pypng
RUN mkdir /source
WORKDIR /source
RUN mkdir /source/app
ADD app /source/app
RUN mkdir /source/mongorepo
ADD mongorepo /source/mongorepo
RUN mkdir /source/mnist
ADD mnist /source/mnist
RUN mkdir /source/models
ADD models /source/models
RUN mkdir /source/services
ADD services /source/services

ADD capsnet-fashion.py logging.conf /source/
CMD python capsnet-fashion.py

