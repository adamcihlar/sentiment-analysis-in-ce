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


source_mall = read_mall().sample(100)
source_facebook = read_facebook().sample(100)
datasets = [source_facebook, source_mall]

train_datasets, val_datasets = get_source_datasets_ready_for_finetuning(
    datasets,
    drop_neutral=True,
    preprocessor=Preprocessor(),
    tokenizer=Tokenizer(),
    batch_size=8,
    shuffle=True,
    num_workers=0,
)

asc = AdaptiveSentimentClassifier(
    Preprocessor(),
    Tokenizer(),
    Encoder(),
    ClassificationHead,
    Discriminator(),
    Encoder(),
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
