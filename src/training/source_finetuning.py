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

if __name__ == "__main__":
    source_mall = read_mall().sample(100)
    source_facebook = read_facebook().sample(100)
    datasets = [source_facebook, source_mall]

    asc = AdaptiveSentimentClassifier(
        Preprocessor(),
        Tokenizer(),
        Encoder(),
        ClassificationHead,
        Discriminator(),
        Encoder(),
    )

    train_datasets, val_datasets = get_datasets_ready_for_finetuning(
        datasets,
        drop_neutral=True,
        preprocessor=asc.preprocessor,
        tokenizer=asc.tokenizer,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    asc.finetune(
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        optimizer=AdamW,
        optimizer_params={"lr": 2e-5, "betas": (0.9, 0.999)},
        lr_decay=0.95,
        lr_scheduler_call=get_linear_schedule_with_warmup,
        warmup_steps_proportion=0.1,
        num_epochs=4,
        metrics=["f1", "accuracy", "precision", "recall"],
    )
