## Sentiment Analysis in Correspondence Experiments

### How to run demo app
0. git clone https://github.com/adamcihlar/sentiment-analysis-in-ce.git

#### With docker:
1. from root of the repo run *make docker_build_app* - builds image, takes longer to install all the requirements
2. then run *make run_app*  - spins up the container
3. the app is running at *localhost:8081*

#### Without docker:
1. requires environment with **python 3.8** installed
2. from root of the repo run *make install*  - installs all the requirements
3. then run *make app*
4. the app is running at *localhost:8081*
