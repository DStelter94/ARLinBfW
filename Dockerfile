# https://github.com/wesnoth/wesnoth/blob/master/utils/dockerbuilds/travis/Dockerfile-base-1804-master

FROM nvidia/cuda:10.2-runtime

ARG USER_ID

RUN apt-get update \
 && apt-get install -y sudo

RUN adduser --disabled-password --gecos '' --uid $USER_ID docker
RUN adduser docker sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# WORKDIR /home/docker

RUN apt update

RUN apt install -y -qq apt-utils

# boost
RUN apt install -y -qq libboost-filesystem1.65-dev libboost-filesystem1.65.1 libboost-iostreams1.65-dev libboost-iostreams1.65.1 libboost-locale1.65-dev libboost-locale1.65.1 libboost-regex1.65-dev libboost-regex1.65.1 libboost-serialization1.65-dev libboost-serialization1.65.1 libasio-dev libboost-program-options1.65-dev libboost-program-options1.65.1 libboost-random1.65-dev libboost-random1.65.1 libboost-system1.65-dev libboost-system1.65.1 libboost-thread1.65-dev libboost-thread1.65.1 libboost-test-dev

# SDL
RUN apt install -y -qq libsdl2-2.0-0 libsdl2-dev libsdl2-image-2.0-0 libsdl2-image-dev libsdl2-mixer-2.0-0 libsdl2-mixer-dev libsdl2-ttf-2.0-0 libsdl2-ttf-dev

# make tzdata not prompt for a timezone
ENV DEBIAN_FRONTEND=noninteractive

# translations
RUN apt install -y -qq asciidoc dos2unix xsltproc po4a docbook-xsl language-pack-en
RUN locale-gen en_US.UTF-8

# misc
RUN apt install -y -qq libpng16-16 libpng-dev libreadline6-dev libvorbis-dev libcairo2 libcairo2-dev libpango-1.0-0 libpango1.0-dev libfribidi0 libfribidi-dev libbz2-1.0 libbz2-dev zlib1g zlib1g-dev libpangocairo-1.0-0 libssl-dev libmysqlclient-dev expect-dev python3-pip moreutils
RUN pip3 install --upgrade pip
RUN yes | pip3 install paramiko

# programs
RUN apt install -y -qq openssl gdb xvfb bzip2 git scons cmake make ccache gcc g++ clang lld

# either use install wesnoth from apt-get or install wesnoth from sources
# install wesnoth from apt-get
RUN apt-get install software-properties-common -y
RUN add-apt-repository ppa:vincent-c/wesnoth
RUN apt-get update
RUN apt-get install wesnoth -y

# install wesnoth from sources
# https://github.com/wesnoth/wesnoth/blob/master/INSTALL.md
# ADD https://sourceforge.net/projects/wesnoth/files/wesnoth-1.14/wesnoth-1.14.13/wesnoth-1.14.13.tar.bz2/download wesnoth-1.14.13.tar.bz2
# RUN tar xvjf wesnoth-1.14.13.tar.bz2
# WORKDIR /wesnoth-1.14.13
# RUN scons
# RUN scons install

# build app
COPY PythonController/requirements.txt /app/requirements.txt
COPY PythonController/gym-bfw /app/gym-bfw
WORKDIR /app
RUN pip3 install -e ./gym-bfw
RUN pip3 install -r requirements.txt

USER docker
ENV PATH="/usr/games/:${PATH}"
RUN wesnoth --userdata-path
RUN mkdir ~/.config/wesnoth-1.14/data/input
RUN mkdir ~/.config/wesnoth-1.14/data/add-ons/PythonAddon

# copy app code and set owner
COPY --chown=docker PythonController /app/
COPY --chown=docker PythonAddon /home/docker/.config/wesnoth-1.14/data/add-ons/PythonAddon/
