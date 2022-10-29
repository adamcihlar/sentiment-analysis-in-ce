FROM python:3.8-buster

RUN apt-get update -y && apt-get install -y \
tmux \
tree \
vim \
&& rm -rf /var/lib/apt/lists/*

# fix encoding issues
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# enable mouse support for tmux
RUN echo " setw -g mouse on" >> ~/.tmux.conf

# I want interactive python environment
RUN pip install ipython

# copy the directory and install libraries
COPY . app
WORKDIR app
RUN make install

