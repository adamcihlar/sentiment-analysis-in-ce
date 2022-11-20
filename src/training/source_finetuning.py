from tqdm import tqdm
import pandas as pd
import numpy as np
from loguru import logger

from sklearn.model_selection import train_test_split
import torch
from torch.optim import AdamW
from transformers import get_scheduler
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datasets import load_metric

from src.reading.readers import read_facebook, read_mall, read_csfd
from src.utils.datasets import (
    ClassificationDataset,
    drop_undefined_classes,
    transform_labels_to_probs,
    get_finetuning_datasets,
)
from src.model.tokenizers import Tokenizer
from src.utils.text_preprocessing import Preprocessor


source_mall = read_mall()
source_facebook = read_facebook()
source_csfd = read_csfd()
datasets = [source_mall, source_facebook, source_csfd]

source_dataset = drop_undefined_classes(source_dataset)
source_dataset = transform_labels_to_probs(source_dataset, drop_neutral=True)

preprocessor = Preprocessor()
tokenizer = Tokenizer()

source_train_df, source_val_df = get_finetuning_datasets(source_dataset)

source_train = ClassificationDataset(
    X=source_train_df.text, y=source_train_df.label, source=source_train_df.source
)
source_train.preprocess(preprocessor)
source_train.tokenize(tokenizer)
source_train.create_dataset()
source_train.create_dataloader(batch_size=1)

next(iter(source_train.torch_dataloader))

list(source_train.source)[0]

source_val = ClassificationDataset(X=source_val_df.text, y=source_val_df.label)
source_val.preprocess(preprocessor)
source_val.tokenize(tokenizer)
source_val.create_dataset()
source_val.create_dataloader()


# load model
model = RobertaForSequenceClassification.from_pretrained(
    "ufal/robeczech-base",
    num_labels=2,
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


# specify training details
optimizer = AdamW(model.parameters(), lr=3e-5)
num_epochs = 1
num_training_steps = num_epochs * len(train_dataloader)
scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=num_training_steps * 0.15,
    num_training_steps=num_training_steps,
)

# gpu if available, else cpu
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# training loop
val_metric = load_metric("f1")  # accuracy
val_metric_progress = []

progress_bar = tqdm(range(num_training_steps))
counter = 0

for epoch in range(num_epochs):

    # training
    model.train()
    loss_progress = []
    for batch in train_dataloader:
        counter += 1
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        loss_progress.append(loss.item())
        if counter % 5 == 0:
            mean_loss = np.mean(np.array(loss_progress))
            print("Training loss:", mean_loss)
            loss_progress = []

    # validation
    model.eval()
    for batch in val_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        val_metric.add_batch(predictions=predictions, references=batch["labels"])

    val_metric_progress.append(val_metric.compute()["f1"])
    print("Validation metric:", val_metric_progress[len(val_metric_progress) - 1])

    save_path = "output/models/finetunned/robeczech_" + str(epoch)
    os.makedirs(os.path.split(save_path)[0], exist_ok=True)
    torch.save(model.state_dict(), save_path)
