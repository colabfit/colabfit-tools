# Docker setup
Write a Dockerfile for setting up the Docker image

```dockerfile
# Contents of 'colabfit/examples/Dockerfile'
FROM ubuntu:20.04

RUN apt-get update
RUN apt-get install -y sudo
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y git
RUN apt-get install -y wget
RUN apt-get install systemctl
```

Build the image
```bash
sudo docker pull ubuntu:20.04
sudo docker build -t colabfit - < Dockerfile
```

Open a terminal inside of a Docker container using the image
```bash
sudo docker run -t -i --entrypoint bash colabfit
```