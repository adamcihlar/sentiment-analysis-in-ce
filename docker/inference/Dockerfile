FROM python:3.8-buster

# fix encoding issues
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# copy the directory
COPY . app
WORKDIR app

# install libraries 
RUN make install

CMD make predictions
