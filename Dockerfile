FROM ubuntu:18.04

MAINTAINER Amazon AI <sage-learner@amazon.com>

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         vim \
         tree \
         build-essential \
         python3-dev \
         libevent-dev \
	     python-all-dev \
         python3-pip \
         python-setuptools \
         python3-setuptools \
         nginx \
         ffmpeg \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s -f /usr/bin/python3 /usr/bin/python
RUN ln -s -f /usr/bin/pip3 /usr/bin/pip
RUN pip3 install --no-cache-dir numpy==1.16.4 \
    scipy==1.0.0  flask statsmodels pydub \
    torch==1.5.0 torchvision==0.6.0 s3fs numba==0.48 tensorflow==1.12.2 \
    tensorboard inflect==0.2.5 matplotlib \
    librosa==0.6.0 Unidecode==1.0.22 pillow boto3 sagemaker gevent ipython

RUN pip3 install gunicorn && \
        rm -rf /root/.cache

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY opt_ml /opt/ml
COPY opt_program2 /opt/program
WORKDIR /opt/program
RUN chmod +x train
RUN chmod +x serve
