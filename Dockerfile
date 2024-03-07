## Build the docker image using:
# docker build --build-arg USERNAME=$USER --build-arg USER_UID --no-cache -t pycsep .

## Docker instructions

# Install Docker image from trusted source
FROM python:3.9.18-slim-bullseye

# Setup user id and permissions.
ARG USERNAME=modeler
ARG USER_UID=1100
ARG USER_GID=$USER_UID
RUN groupadd --non-unique -g $USER_GID $USERNAME \
    && useradd -u $USER_UID -g $USER_GID -s /bin/sh -m $USERNAME

# Install git
RUN apt update
RUN apt install -y git

# Set up work directory in the Docker container.
WORKDIR /usr/src/

# Set up and create python virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install pycsep.
RUN git clone https://github.com/SCECcode/pycsep.git
RUN pip install --no-cache-dir --upgrade pip
RUN cd pycsep && pip install .

# Docker can now be initialized as user
USER $USERNAME

## Run the docker image in interactive mode from the command line
# docker run -it --rm --name pycsep pycsep:latest

