from typing import List, Type, Callable, Dict
import itertools
import random
import math
import time
import os
import json
from tqdm import tqdm

import numpy as np
import torch
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    RobertaForSequenceClassification,
)
from torch.utils.data import DataLoader
from evaluate import load

from src.utils.datasets import Dataset
from src.utils.optimization import layer_wise_learning_rate
from src.model.encoders import Encoder
from src.utils.text_preprocessing import Preprocessor
from src.model.tokenizers import Tokenizer
from src.config import paths


class ClassificationHead(torch.nn.Module):
    """
    Sentiment classifier with standard architecture.
    The output is probabilities, so it is needed to use BCELoss as loss
    function.
    If this was numerically unstable try skipping the sigmoid activation
    function and use BCEWithLogitsLoss instead.
    """

    def __init__(
        self,
        input_size=768,
        hidden_size=768,
        num_classes=1,
        dropout=0.1,
        model=None,
        path_to_finetuned=None,
    ):
        super(ClassificationHead, self).__init__()
        if model is not None:
            self.model = model
        else:
            self.model = torch.nn.Sequential(
                # torch.nn.Dropout(dropout),
                # torch.nn.Linear(input_size, hidden_size),
                # torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_size, num_classes),
                torch.nn.Sigmoid(),
            )
        if path_to_finetuned is not None:
            self.model.load_state_dict(torch.load(path_to_finetuned))

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs


class Discriminator(torch.nn.Module):
    """
    Classifier trained to distinguish between source and target domain.
    """

    def __init__(
        self, input_size=768, hidden_size=3072, num_classes=1, dropout=0.0, model=None
    ):
        super(Discriminator, self).__init__()
        if model is not None:
            self.model = model
        else:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(input_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, num_classes),
                torch.nn.Sigmoid(),
            )

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs


class AdaptiveSentimentClassifier:
    def __init__(
        self,
        preprocessor,
        tokenizer,
        source_encoder,
        classifier: Type[ClassificationHead],
        discriminator,
        target_encoder=None,
    ):
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.source_encoder = source_encoder
        self.classifier = classifier
        self.target_encoder = target_encoder
        self.discriminator = discriminator
        self.name = self.source_encoder.name

    def finetune(
        self,
        train_datasets: Dict,
        val_datasets: Dict,
        optimizer: Type[torch.optim.Optimizer],
        optimizer_params: Dict,
        lr_params: Dict,
        lr_scheduler_call: Callable,
        warmup_steps_proportion: float,
        num_epochs: int,
        metrics: List,
    ):
        # save start time of the training
        start_time = time.strftime("%Y%m%d-%H%M%S")

        # get one classfication head per dataset and shared encoder for all datasets
        classifiers = {ds_name: self.classifier() for ds_name in train_datasets}
        encoder = self.source_encoder

        # get optimizer for each classfication head and the shared encoder
        cls_optimizers = {
            cls_name: optimizer(classifiers[cls_name].parameters(), **optimizer_params)
            for cls_name in classifiers
        }
        if lr_params.get("lr_decay") is None or lr_params.get("lr_decay") == 1:
            encoder_optimizer = optimizer(encoder.parameters(), **optimizer_params)
        else:
            list_of_layers = encoder.encoder.encoder.layer
            optimizer_params_list = layer_wise_learning_rate(
                list_of_layers, optimizer_params["lr"], **lr_params
            )
            encoder_optimizer = optimizer(optimizer_params_list, **optimizer_params)

        # compute the total num of training steps to init lr_scheduler
        num_steps_per_epoch_per_dataloader = {
            ds_name: len(train_datasets[ds_name].torch_dataloader)
            for ds_name in train_datasets
        }
        num_training_steps = (
            sum(num_steps_per_epoch_per_dataloader.values()) * num_epochs
        )

        # get sequence of how batches will be taken from dataloaders
        dl_samples = [
            [dl] * num_steps_per_epoch_per_dataloader[dl]
            for dl in num_steps_per_epoch_per_dataloader
        ]
        samples_schedule = [ds for sublist in dl_samples for ds in sublist]
        random.shuffle(samples_schedule)

        # get a lr_scheduler for every optimizer
        all_optimizers_list = list(cls_optimizers.values()) + [encoder_optimizer]
        lr_schedulers = [
            lr_scheduler_call(
                optimizer=optim,
                num_warmup_steps=num_training_steps * warmup_steps_proportion,
                num_training_steps=num_training_steps,
            )
            for optim in all_optimizers_list
        ]

        # put all models to device, gpu if available, else cpu
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        [cls.to(device) for cls in classifiers.values()]
        encoder.to(device)

        # track training info
        val_metrics = {metric: load(metric) for metric in metrics}
        val_metrics_progress = {ds: {} for ds in val_datasets}
        for ds in val_metrics_progress:
            val_metrics_progress[ds] = {metric_name: [] for metric_name in val_metrics}
        train_loss_mean_progress = {ds_name: [] for ds_name in train_datasets}
        val_loss_mean_progress = {ds_name: [] for ds_name in val_datasets}
        train_loss_batch_progress = {ds_name: [] for ds_name in train_datasets}

        # display training progress
        progress_bar = tqdm(range(num_training_steps))
        counter = 0
        display_loss_after_iters = math.ceil(
            sum(num_steps_per_epoch_per_dataloader.values()) / 100
        )
        # define training loss
        bce_loss = torch.nn.BCELoss()

        # training loop
        for epoch in range(num_epochs):

            # reshuffle the sampling schedule
            random.shuffle(samples_schedule)
            encoder.train()
            [classifiers[cls_name].train() for cls_name in classifiers]
            loss_progress = []
            train_epoch_loss_progress = {ds_name: [] for ds_name in train_datasets}

            dataloader_iterators = {
                ds_name: iter(train_datasets[ds_name].torch_dataloader)
                for ds_name in train_datasets
            }

            for source_ds in samples_schedule:
                # get batch from the selected data source
                batch = {
                    k: v.to(device)
                    for k, v in next(dataloader_iterators[source_ds]).items()
                }
                # forward pass
                features = encoder(**batch)
                predictions = classifiers[source_ds].forward(features)
                # backward pass
                cls_loss = bce_loss(
                    predictions, batch["labels"].unsqueeze(1).to(torch.float32)
                )
                cls_loss.backward()
                # optimize the corresponding classifier and encoder
                cls_optimizers[source_ds].step()
                encoder_optimizer.step()
                # update learning rates for all optimizers
                [lr_scheduler.step() for lr_scheduler in lr_schedulers]
                # reset gradients
                cls_optimizers[source_ds].zero_grad()
                encoder_optimizer.zero_grad()
                progress_bar.update(1)

                loss_progress.append(cls_loss.item())
                train_epoch_loss_progress[source_ds].append(cls_loss.item())
                if counter % display_loss_after_iters == 0:
                    mean_loss = np.mean(np.array(loss_progress))
                    print("Training loss:", mean_loss)
                    loss_progress = []

            # validation
            encoder.eval()
            [classifiers[cls_name].eval() for cls_name in classifiers]
            val_epoch_loss_progress = {ds_name: [] for ds_name in val_datasets}

            for val_ds_name in val_datasets:
                for batch in val_datasets[val_ds_name].torch_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.no_grad():
                        features = encoder(**batch)
                        probs = classifiers[val_ds_name].forward(features)

                    cls_loss = bce_loss(
                        probs, batch["labels"].unsqueeze(1).to(torch.float32)
                    )
                    val_epoch_loss_progress[val_ds_name].append(cls_loss.item())
                    predictions = torch.round(probs)
                    [
                        val_metrics[val_metric].add_batch(
                            predictions=predictions, references=batch["labels"]
                        )
                        for val_metric in val_metrics
                    ]
                [
                    val_metrics_progress[val_ds_name][val_metric].append(
                        val_metrics[val_metric].compute()[val_metric]
                    )
                    for val_metric in val_metrics
                ]

            # compute average losses per epoch
            for ds_name in train_datasets:
                train_loss_mean_progress[ds_name].append(
                    np.mean(np.array(train_epoch_loss_progress[ds_name]))
                )
                train_loss_batch_progress[ds_name].append(
                    train_epoch_loss_progress[ds_name]
                )
            train_epoch_loss_progress = {ds_name: [] for ds_name in train_datasets}
            for ds_name in val_datasets:
                val_loss_mean_progress[ds_name].append(
                    np.mean(np.array(val_epoch_loss_progress[ds_name]))
                )
            val_epoch_loss_progress = {ds_name: [] for ds_name in val_datasets}

            # save all models from the epoch
            # encoder
            save_path = os.path.join(
                paths.OUTPUT_MODELS_FINETUNNED_ENCODER,
                "_".join([self.name, start_time, str(epoch)]),
            )
            os.makedirs(os.path.split(save_path)[0], exist_ok=True)
            torch.save(encoder.state_dict(), save_path)

            # classifiers
            cls_save_paths = [
                os.path.join(
                    paths.OUTPUT_MODELS_FINETUNNED_CLASSIFIER,
                    "_".join([self.name, start_time, cls, str(epoch)]),
                )
                for cls in classifiers
            ]
            [
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                for path in cls_save_paths
            ]
            [
                torch.save(classifiers[cls_name].state_dict(), path)
                for cls_name, path in zip(classifiers, cls_save_paths)
            ]

        for ds_name in val_datasets:
            val_metrics_progress[ds_name]["val_loss"] = val_loss_mean_progress[ds_name]
            val_metrics_progress[ds_name]["train_loss"] = train_loss_mean_progress[
                ds_name
            ]
        info_save_path = os.path.join(
            paths.OUTPUT_INFO_FINETUNNING, "_".join(self.name, start_time)
        )
        with open(info_save_path, "w") as fp:
            json.dump(val_metrics_progress, fp)
        pass


if __name__ == "__main__":

    asc = AdaptiveSentimentClassifier(
        Preprocessor(), Tokenizer(), Encoder(), ClassificationHead, Discriminator()
    )

    # finetuning args
    source_encoder = asc.source_encoder
    train_datasets
    val_datasets
    optimizer = AdamW
    optimizer_params = {"lr": 2e-5, "betas": (0.9, 0.999)}
    layer_wise_lr_decay = 0.95
    n_layers_following = 1  # n layers of the classifier
    warmup_steps_proportion = 0.1
    num_epochs = 1
    metrics = ["f1", "accuracy", "precision", "recall"]

    # method body
    # save start time of the training
    start_time = time.strftime("%Y%m%d-%H%M%S")

    # get one classfication head per dataset and shared encoder for all datasets
    classifiers = {ds_name: asc.classifier() for ds_name in train_datasets}
    encoder = source_encoder

    # get optimizer for each classfication head and the shared encoder
    cls_optimizers = {
        cls_name: optimizer(classifiers[cls_name].parameters(), **optimizer_params)
        for cls_name in classifiers
    }
    if layer_wise_lr_decay is None or layer_wise_lr_decay == 1:
        encoder_optimizer = optimizer(encoder.parameters(), **optimizer_params)
    else:
        list_of_layers = source_encoder.encoder.encoder.layer
        optimizer_params_list = layer_wise_learning_rate(
            list_of_layers, optimizer_params["lr"], layer_wise_lr_decay
        )
        encoder_optimizer = optimizer(optimizer_params_list, **optimizer_params)

    # compute the total num of training steps to init lr_scheduler
    num_steps_per_epoch_per_dataloader = {
        ds_name: len(train_datasets[ds_name].torch_dataloader)
        for ds_name in train_datasets
    }
    num_training_steps = sum(num_steps_per_epoch_per_dataloader.values()) * num_epochs

    # get sequence of how batches will be taken from dataloaders
    dl_samples = [
        [dl] * num_steps_per_epoch_per_dataloader[dl]
        for dl in num_steps_per_epoch_per_dataloader
    ]
    samples_schedule = [ds for sublist in dl_samples for ds in sublist]
    random.shuffle(samples_schedule)

    # get a lr_scheduler for every optimizer
    all_optimizers_list = list(cls_optimizers.values()) + [encoder_optimizer]
    lr_schedulers = [
        get_linear_schedule_with_warmup(
            optimizer=optim,
            num_warmup_steps=num_training_steps * warmup_steps_proportion,
            num_training_steps=num_training_steps,
        )
        for optim in all_optimizers_list
    ]

    # put all models to device, gpu if available, else cpu
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    [cls.to(device) for cls in classifiers.values()]
    encoder.to(device)

    # track training info
    val_metrics = {metric: load(metric) for metric in metrics}
    val_metrics_progress = {ds: {} for ds in val_datasets}
    for ds in val_metrics_progress:
        val_metrics_progress[ds] = {metric_name: [] for metric_name in val_metrics}
    train_loss_mean_progress = {ds_name: [] for ds_name in train_datasets}
    val_loss_mean_progress = {ds_name: [] for ds_name in val_datasets}
    train_loss_batch_progress = {ds_name: [] for ds_name in train_datasets}

    # display training progress
    progress_bar = tqdm(range(num_training_steps))
    counter = 0
    display_loss_after_iters = math.ceil(
        sum(num_steps_per_epoch_per_dataloader.values()) / 100
    )
    # define training loss
    bce_loss = torch.nn.BCELoss()

    # training loop
    for epoch in range(num_epochs):

        # reshuffle the sampling schedule
        random.shuffle(samples_schedule)
        encoder.train()
        [classifiers[cls_name].train() for cls_name in classifiers]
        loss_progress = []
        train_epoch_loss_progress = {ds_name: [] for ds_name in train_datasets}

        dataloader_iterators = {
            ds_name: iter(train_datasets[ds_name].torch_dataloader)
            for ds_name in train_datasets
        }

        for source_ds in samples_schedule:
            # get batch from the selected data source
            batch = {
                k: v.to(device)
                for k, v in next(dataloader_iterators[source_ds]).items()
            }
            # forward pass
            features = encoder(**batch)
            predictions = classifiers[source_ds].forward(features)
            # backward pass
            cls_loss = bce_loss(
                predictions, batch["labels"].unsqueeze(1).to(torch.float32)
            )
            cls_loss.backward()
            # optimize the corresponding classifier and encoder
            cls_optimizers[source_ds].step()
            encoder_optimizer.step()
            # update learning rates for all optimizers
            [lr_scheduler.step() for lr_scheduler in lr_schedulers]
            # reset gradients
            cls_optimizers[source_ds].zero_grad()
            encoder_optimizer.zero_grad()
            progress_bar.update(1)

            loss_progress.append(cls_loss.item())
            train_epoch_loss_progress[source_ds].append(cls_loss.item())
            if counter % display_loss_after_iters == 0:
                mean_loss = np.mean(np.array(loss_progress))
                print("Training loss:", mean_loss)
                loss_progress = []

        # validation
        encoder.eval()
        [classifiers[cls_name].eval() for cls_name in classifiers]
        val_epoch_loss_progress = {ds_name: [] for ds_name in val_datasets}

        for val_ds_name in val_datasets:
            for batch in val_datasets[val_ds_name].torch_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    features = encoder(**batch)
                    probs = classifiers[val_ds_name].forward(features)

                cls_loss = bce_loss(
                    probs, batch["labels"].unsqueeze(1).to(torch.float32)
                )
                val_epoch_loss_progress[val_ds_name].append(cls_loss.item())
                predictions = torch.round(probs)
                [
                    val_metrics[val_metric].add_batch(
                        predictions=predictions, references=batch["labels"]
                    )
                    for val_metric in val_metrics
                ]
            [
                val_metrics_progress[val_ds_name][val_metric].append(
                    val_metrics[val_metric].compute()[val_metric]
                )
                for val_metric in val_metrics
            ]

        # compute average losses per epoch
        for ds_name in train_datasets:
            train_loss_mean_progress[ds_name].append(
                np.mean(np.array(train_epoch_loss_progress[ds_name]))
            )
            train_loss_batch_progress[ds_name].append(
                train_epoch_loss_progress[ds_name]
            )
        train_epoch_loss_progress = {ds_name: [] for ds_name in train_datasets}
        for ds_name in val_datasets:
            val_loss_mean_progress[ds_name].append(
                np.mean(np.array(val_epoch_loss_progress[ds_name]))
            )
        val_epoch_loss_progress = {ds_name: [] for ds_name in val_datasets}

        # save all models from the epoch
        # encoder
        save_path = os.path.join(
            paths.OUTPUT_MODELS_FINETUNNED_ENCODER,
            "_".join([asc.name, start_time, str(epoch)]),
        )
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
        torch.save(encoder.state_dict(), save_path)

        # classifiers
        cls_save_paths = [
            os.path.join(
                paths.OUTPUT_MODELS_FINETUNNED_CLASSIFIER,
                "_".join([asc.name, start_time, cls, str(epoch)]),
            )
            for cls in classifiers
        ]
        [os.makedirs(os.path.split(path)[0], exist_ok=True) for path in cls_save_paths]
        [
            torch.save(classifiers[cls_name].state_dict(), path)
            for cls_name, path in zip(classifiers, cls_save_paths)
        ]

    for ds_name in val_datasets:
        val_metrics_progress[ds_name]["val_loss"] = val_loss_mean_progress[ds_name]
        val_metrics_progress[ds_name]["train_loss"] = train_loss_mean_progress[ds_name]

    info_save_path = os.path.join(
        OUTPUT_INFO_FINETUNNING, "_".join([asc.name, start_time])
    )
    with open(info_save_path, "w") as fp:
        json.dump(val_metrics_progress, fp)
