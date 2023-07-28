# Sentiment Analysis in Correspondence Experiments

## How to run the finetuned sentiment model (without target adaptation)
0. `git clone https://github.com/adamcihlar/sentiment-analysis-in-ce.git`

### Interactive app
#### With docker:
1. from root of the repo run `make run_app` - builds image and spins up the container
2. the app is running at *localhost:8081*

#### Without docker:
1. requires environment with **python 3.8** installed
2. from root of the repo run `make install`  - installs all the requirements
3. then run `make app`
4. the app is running at *localhost:8081*

### API
#### With docker:
1. from root of the repo run `make run_api` - builds image and spins up the container
2. the API is running at *localhost:8081*

#### Without docker:
1. requires environment with **python 3.8** installed
2. from root of the repo run `make install`  - installs all the requirements
3. then run `make api`
4. the app is running at *localhost:8081*

### Bulk predictions
#### With docker:
1. store the texts that should be evaluated in folder `input/` in `.csv` format (first row is a column name)
2. from root of the repo run `make run_predictions` - builds image and spins up the container and runs the inference
3. the predictions are saved at `output/predictions/sentiment_predictions.csv`

#### Without docker:
1. requires environment with **python 3.8** installed
2. store the texts that should be evaluated in folder `input/` in `.csv` format (first row is a column name)
3. from root of the repo run `make install`  - installs all the requirements
4. then run `make predictions`
5. the predictions are saved at `output/predictions/sentiment_predictions.csv`

## How to run the sentiment model with support of labelled subset
0. `git clone https://github.com/adamcihlar/sentiment-analysis-in-ce.git`
### Bulk predictions
#### With docker:
1. store the texts that should be evaluated in folder `input/` in `.csv` format (first row is a column name)
2. from root of the repo run `make run_supported_predictions` - builds image and spins up the container and runs the inference
3. follow the instructions in the command line
4. the predictions are saved at `output/predictions/supported_sentiment_predictions.csv`

#### Without docker:
1. requires environment with **python 3.8** installed
2. store the texts that should be evaluated in folder `input/` in `.csv` format (first row is a column name)
3. from root of the repo run `make install`  - installs all the requirements
4. then run `make supported_predictions`
5. follow the instructions in the command line
6. the predictions are saved at `output/predictions/supported_sentiment_predictions.csv`
