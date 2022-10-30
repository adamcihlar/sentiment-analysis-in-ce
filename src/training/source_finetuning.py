from tqdm import tqdm
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import torch
from torch.optim import AdamW
from transformers import get_scheduler
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datasets import load_metric

from src.utils.dataset import Dataset
from src.reading.readers import read_facebook


df = read_facebook()
df = df.loc[df.label.isin([0, 1])]

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_test, y_test, test_size=0.5, random_state=42
)

tokenizer = RobertaTokenizer.from_pretrained("ufal/robeczech-base")

# tokenize texts
X_train_tok = tokenizer(
    list(X_train), padding="max_length", max_length=512, truncation=True
)
X_val_tok = tokenizer(
    list(X_val), padding="max_length", max_length=512, truncation=True
)
X_test_tok = tokenizer(
    list(X_test), padding="max_length", max_length=512, truncation=True
)

# load data to torch Datasets
train_dataset = Dataset(X_train_tok, list(y_train))
val_dataset = Dataset(X_val_tok, list(y_val))
test_dataset = Dataset(X_test_tok, list(y_test))

# create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=16, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=16, num_workers=2)

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
