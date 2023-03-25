## Sentiment Analysis in Correspondence Experiments

### How to run demo app
0. `git clone https://github.com/adamcihlar/sentiment-analysis-in-ce.git`

#### With docker:
1. from root of the repo run `make docker_build_app` - builds image, takes longer to install all the requirements
2. then run `make run_app`  - spins up the container
3. the app is running at *localhost:8081*

#### Without docker:
1. requires environment with **python 3.8** installed
2. from root of the repo run `make install`  - installs all the requirements
3. then run `make app`
4. the app is running at *localhost:8081*

#### Devel
Scenario:
    * I have one target dataset w/o labels and multiple source datasets w/ labels that share the
      label space with the target
    * The goal is to get a model that will be performing the best on the target
1. Get source training data with labels
2. Split to train-val-test
3. Finetune model(s)
4. Evaluate them on test set(s) and save the best score - this is your `golden
   standard`
5. Pick a model (or decide how to pick it) that will be starting point for the latter adaptation - in general models that have the smallest **performance drop** (`golden standard - metric for the model`) on the unseen target. This decision is whether to use independent or shared encoders and classifiers, in case you go for independent, you have to choose the particular model, if you go with shared, take the encoder/classifier/both trained on all sources together.
6. Decide on the adaptation parameters (epochs, temp, loss combination) - this is still an open question, should I take what worked overall or what worked for the model (type) selected
7. Few-shot adaptation?
