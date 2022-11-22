from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.reading.readers import read_facebook, read_mall, read_csfd
from src.utils.datasets import get_source_datasets_ready_for_finetuning
from src.utils.text_preprocessing import Preprocessor
from src.model.tokenizers import Tokenizer
from src.model.encoders import Encoder
from src.model.classifiers import (
    AdaptiveSentimentClassifier,
    ClassificationHead,
    Discriminator,
)


source_mall = read_mall().sample(12)
source_facebook = read_facebook().sample(13)
datasets = [source_facebook, source_mall]

train_datasets, val_datasets = get_source_datasets_ready_for_finetuning(
    datasets,
    drop_neutral=True,
    preprocessor=Preprocessor(),
    tokenizer=Tokenizer(),
    batch_size=4,
    shuffle=True,
    num_workers=0,
)

asc = AdaptiveSentimentClassifier(
    Preprocessor(), Tokenizer(), Encoder(), ClassificationHead, Discriminator()
)

asc.finetune(
    train_datasets=train_datasets,
    val_datasets=val_datasets,
    optimizer=AdamW,
    optimizer_params={"lr": 2e-5, "betas": (0.9, 0.999)},
    lr_params={"lr_decay": 0.95, "n_layers_following": 1},
    lr_scheduler_call=get_linear_schedule_with_warmup,
    warmup_steps_proportion=0.1,
    num_epochs=4,
    metrics=["f1", "accuracy", "precision", "recall"],
)


### training details from the RobeCzech paper

# Data:
# Facebook sentiment dataset, bipolar ignored
# Since I want just a scale [negative - positive] I will not take the neutral
# samples into consideration - I won't be able to compare my results with their
# so I will just take their training settings
# 10-fold cross validation (train, test split) and validation set to determine
# the best learning rate, it seems that they always use the same number of epochs

# Architecture:
# "Standard text classification architecture"
# One softmax layer on top of the encoder - will be logistic sigmoid for me

# Optimization:
# lazy Adam optimizer = torch.optim.SparseAdam
# batch size = 64
# 1. epoch - only classifier is trained, lr=1e-3
# 2.-5. epochs - whole model is trained, lr cosine warm up from zero to 3e-5
# 6.-15. epochs - whole model is trained, lr cosine decay back to zero

# In paper:
# How to Fine-Tune BERT for Text Classification?
# 1. Take the last layer as embeddings
# 2. If sequence is longer than 512 tokens, take first 128 and last 382 - would be
# 3. batch size 24
# 4. dropout 0.1
# 5. Adam optimizer (I will use AdamW as it should generalize better) with
# b1=0.9 and b2=0.999
# learning rate = 2e-5
# 4 epochs
