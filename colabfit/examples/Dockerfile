FROM ubuntu:20.04

RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y git
RUN pip install git+https://github.com/colabfit/colabfit-tools.git@master
RUN apt-get install -y wget
RUN wget -qO - https://www.mongodb.org/static/pgp/server-5.0.asc | apt-key add -
RUN echo "deb [ arch=amd64,arm64   ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/5.0 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-5.0.list
RUN apt-get update
RUN apt-get install -y mongodb-org
RUN apt-get install systemctl
RUN systemctl start mongod
