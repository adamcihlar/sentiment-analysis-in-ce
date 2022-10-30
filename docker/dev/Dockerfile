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

# copy the directory
COPY . app
WORKDIR app

# I want interactive python environment
RUN pip install ipython
RUN echo 'alias ipython="ipython --no-autoindent"' >> ~/.bashrc

# install libraries 
RUN make install

