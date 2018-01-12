FROM ubuntu:16.04
MAINTAINER Wolfgang Giersche
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
RUN pip3 install pymongo flask
RUN mkdir /source
WORKDIR /source
RUN mkdir /source/app
RUN mkdir /source/mongorepo
ADD app /source/app
ADD mongorepo /source/mongorepo
ADD capsnet-fashion.py logging.conf /source/
CMD python3 capsnet-fashion.py

