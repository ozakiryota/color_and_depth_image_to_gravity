FROM nvidia/cuda:10.0-devel-ubuntu18.04

########## basis ##########
RUN apt-get update && \
	apt-get install -y \
		vim \
		wget \
		unzip \
		git \
		build-essential
######### Python ##########
RUN apt-get update &&\
	apt-get install -y \
		python3.6 \
		python3-pip &&\
	pip3 install \
		tqdm \
		matplotlib \
		tensorflow \
		tensorboardX \
		torch==1.0.1 torchvision==0.2.2
		# torch==1.2.0 torchvision==0.4.0
######### NO cache ##########
ARG CACHEBUST=1
######### My package ##########
RUN cd /home &&\
	git clone https://github.com/ozakiryota/color_and_depth_image_to_gravity
######### initial position ##########
WORKDIR /home/color_and_depth_image_to_gravity/pysrc
