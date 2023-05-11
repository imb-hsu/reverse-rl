#FROM python:3.8
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

#install python basics´
RUN apt-get update && \
    apt-get install -y \
        python3.8-dev \
        python3.8-distutils \
        python3.8-venv \
        python3-pip \
        python3-setuptools \
        git \
        wget \
        unzip \
        vim \
        sudo && \
    rm -rf /var/lib/apt/lists/*


# set a directory for the app
WORKDIR /Project/

# copy all the files to the container
COPY ./python ./code
COPY ./build_beaming ./build_beaming
COPY ./requirements.txt ./requirements.txt

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install --no-deps gym_unity==0.28.0
 
# add shared volume
VOLUME /Logging

# run the command
ENTRYPOINT ["python3"]
CMD ["./code/main.py", "PPO", "5", "10000", "3", "100" ]