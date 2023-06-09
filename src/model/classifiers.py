from typing import List, Type, Callable, Dict
import itertools
import random
import math
import time
import os
import json
from tqdm import tqdm
from loguru import logger

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from evaluate import load

from src.utils.datasets import ClassificationDataset
from src.utils.optimization import (
    layer_wise_learning_rate,
    inverted_sigmoid,
    to_cuda,
    get_log_prob_for_kl_div,
)

from src.model.encoders import Encoder
from src.utils.text_preprocessing import Preprocessor
from src.model.tokenizers import Tokenizer
from src.config import paths
from src.config.parameters import ClassifierParams, DiscriminatorParams


class ClassificationHead(torch.nn.Module):
    """
    Sentiment classifier with standard architecture.
    The output is probabilities, so it is needed to use BCELoss as loss
    function.
    If this was numerically unstable try skipping the sigmoid activation
    function and use BCEWithLogitsLoss instead.
    Layers wrapped to Sequential modules to enable iterating over them.
    """

    def __init__(
        self,
        input_size=ClassifierParams.INPUT_SIZE,
        hidden_size=ClassifierParams.HIDDEN_SIZE,
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
                torch.nn.Sequential(
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(input_size, hidden_size),
                    torch.nn.Tanh(),
                ),
                torch.nn.Sequential(
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden_size, num_classes),
                    torch.nn.Sigmoid(),
                ),
            )
        if path_to_finetuned is not None:
            logger.info(
                f"Loading model parameters for ClassificationHead from {path_to_finetuned}."
            )
            self.load_state_dict(torch.load(path_to_finetuned))

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs


class Discriminator(torch.nn.Module):
    """
    Classifier trained to distinguish between source and target domain.
    Layers wrapped to Sequential modules to enable iterating over them.
    What sizes of the layers?:
        Original ADDA paper says input, 1024, 2048, output
        Distilation paper says input, 3072, 3072, output
    """

    def __init__(
        self,
        input_size=DiscriminatorParams.INPUT_SIZE,
        hidden_size=DiscriminatorParams.HIDDEN_SIZE,
        num_classes=1,
        dropout=0.0,
        model=None,
    ):
        super(Discriminator, self).__init__()
        if model is not None:
            self.model = model
        else:
            self.model = torch.nn.Sequential(
                torch.nn.Sequential(
                    torch.nn.Linear(input_size, hidden_size),
                    torch.nn.ReLU(),
                ),
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_size, hidden_size),
                    torch.nn.ReLU(),
                ),
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_size, num_classes),
                    torch.nn.Sigmoid(),
                ),
            )
        pass

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
        target_encoder,
        classifier_checkpoint_path=None,
        inference_mode=False,
    ):
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        if inference_mode:
            self.source_encoder = None
            self.name = self.target_encoder.name
        else:
            self.source_encoder = source_encoder
            self.discriminator = discriminator
            self.name = self.source_encoder.name
        if classifier_checkpoint_path is not None:
            self.classifier = classifier(path_to_finetuned=classifier_checkpoint_path)
        else:
            self.classifier = classifier
        self.target_encoder = target_encoder

    def finetune(
        self,
        train_datasets: Dict,
        val_datasets: Dict,
        optimizer: Type[torch.optim.Optimizer],
        optimizer_params: Dict,
        lr_decay: float,
        lr_scheduler_call: Callable,
        warmup_steps_proportion: float,
        num_epochs: int,
        metrics: List,
    ):
        """
        Implements multitask finetuning on provided train_datasets.
        Hyperparameters taken from paper: How to Fine-Tune BERT for Text Classification?
        1. Take the last layer as embeddings
        2. If sequence is longer than 512 tokens, take first 128 and last 382 - would be
        3. batch size 24
        4. dropout 0.1
        5. Adam optimizer (I will use AdamW as it should generalize better) with
        b1=0.9 and b2=0.999
        learning rate = 2e-5
        6. layer wise learning rate decay
        7. 4 epochs

        Saves:
            all the trained models every epoch - one classifier per dataset and
            one shared encoder
            training info - train_loss, val_loss, val_metrics
        """
        # save start time of the training
        start_time = time.strftime("%Y%m%d-%H%M%S")

        # get one classfication head per dataset and shared encoder for all datasets
        classifiers = {ds_name: self.classifier() for ds_name in train_datasets}
        encoder = self.source_encoder

        # get optimizer for each classfication head and the shared encoder
        if lr_decay is None or lr_decay == 1:
            cls_optimizers = {
                cls_name: optimizer(
                    classifiers[cls_name].parameters(), **optimizer_params
                )
                for cls_name in classifiers
            }
            encoder_optimizer = optimizer(encoder.parameters(), **optimizer_params)
        else:
            cls_lists_of_layers = {
                cls_name: list(classifiers[cls_name].model) for cls_name in classifiers
            }
            cls_optimizer_params_lists = {
                cls_name: layer_wise_learning_rate(
                    cls_lists_of_layers[cls_name], optimizer_params["lr"], lr_decay, 0
                )
                for cls_name in cls_lists_of_layers
            }
            cls_optimizers = {
                cls_name: optimizer(
                    cls_optimizer_params_lists[cls_name], **optimizer_params
                )
                for cls_name in classifiers
            }

            encoder_list_of_layers = encoder.encoder.encoder.layer
            optimizer_params_list = layer_wise_learning_rate(
                encoder_list_of_layers,
                optimizer_params["lr"],
                lr_decay,
                len(self.classifier().model),
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
        counter = 1
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
                # get batch from the selected data source and put to the right device
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
                counter += 1

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
                epoch_train_loss = np.mean(np.array(train_epoch_loss_progress[ds_name]))
                train_loss_mean_progress[ds_name].append(epoch_train_loss)
                train_loss_batch_progress[ds_name].append(
                    train_epoch_loss_progress[ds_name]
                )
                print(
                    f"Mean training loss for {ds_name} for epoch {epoch}: {epoch_train_loss}"
                )
            train_epoch_loss_progress = {ds_name: [] for ds_name in train_datasets}
            for ds_name in val_datasets:
                epoch_val_loss = np.mean(np.array(val_epoch_loss_progress[ds_name]))
                val_loss_mean_progress[ds_name].append(epoch_val_loss)
                print(
                    f"Mean validation loss for {ds_name} for epoch {epoch}: {epoch_val_loss}"
                )
            val_epoch_loss_progress = {ds_name: [] for ds_name in val_datasets}

            # save all models from the epoch
            # encoder
            save_path = os.path.join(
                paths.OUTPUT_MODELS_FINETUNED_ENCODER,
                "_".join([self.name, start_time, str(epoch)]),
            )
            os.makedirs(os.path.split(save_path)[0], exist_ok=True)
            torch.save(encoder.state_dict(), save_path)

            # classifiers
            cls_save_paths = [
                os.path.join(
                    paths.OUTPUT_MODELS_FINETUNED_CLASSIFIER,
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

        # save the training info
        for ds_name in val_datasets:
            val_metrics_progress[ds_name]["val_loss"] = val_loss_mean_progress[ds_name]
            val_metrics_progress[ds_name]["train_loss"] = train_loss_mean_progress[
                ds_name
            ]
        info_save_path = os.path.join(
            paths.OUTPUT_INFO_FINETUNING,
            "_".join([self.name, "val", start_time]) + ".json",
        )
        os.makedirs(os.path.split(info_save_path)[0], exist_ok=True)
        with open(info_save_path, "w+") as fp:
            json.dump(val_metrics_progress, fp)

        info_save_path = os.path.join(
            paths.OUTPUT_INFO_FINETUNING,
            "_".join([self.name, "train", start_time]) + ".json",
        )
        os.makedirs(os.path.split(info_save_path)[0], exist_ok=True)
        with open(info_save_path, "w+") as fp:
            json.dump(train_loss_batch_progress, fp)

        # save datasets as I will need them for adapt method
        for ds_name in train_datasets:
            save_path = os.path.join(
                paths.DATA_FINAL_SOURCE_TRAIN,
                "_".join([self.name, start_time, ds_name]) + ".csv",
            )
            os.makedirs(os.path.split(save_path)[0], exist_ok=True)
            train_datasets[ds_name].save_data(save_path)
        for ds_name in val_datasets:
            save_path = os.path.join(
                paths.DATA_FINAL_SOURCE_VAL,
                "_".join([self.name, start_time, ds_name]) + ".csv",
            )
            os.makedirs(os.path.split(save_path)[0], exist_ok=True)
            val_datasets[ds_name].save_data(save_path)

        pass

    def adapt(
        self,
        source_train,
        source_val,
        target,
        optimizer,
        optimizer_params,
        lr_decay,
        lr_scheduler_call,
        warmup_steps_proportion,
        num_epochs,
        temperature,  # for knowledge distilation
        loss_combination_params,  # tuple with alpha and beta
        metrics,
    ):
        """
        Implements adversarial domain adaptation with distilation.
        Hyperparameters:
            How to finetune BERT?:
                1. Take the last layer as embeddings
                2. If sequence is longer than 512 tokens, take first 128 and last 382 - would be
                3. batch size 24
                4. dropout 0.1
                5. Adam optimizer (I will use AdamW as it should generalize better) with
                b1=0.9 and b2=0.999
                6. learning rate = 2e-5
                7. layer wise learning rate decay with 0.95
                8. 4 epochs
            Knowledge Distillation for BERT Unsupervised Domain Adaptation:
                1. ?
                2. ?
                3. batch size 64
                4. ?
                5. Adam optimizer (I will use AdamW as it should generalize better) with
                b1=0.9 and b2=0.999
                6. learning rate = 1e-5
                7. None
                8. 3 epochs
            My settings:
                1. I have to go with the same as in finetuning since the
                classifier is trained on that
                2. Same as 1.
                3. Considerable - but maybe rather 24
                4. No reason to change it from finetuning - keep it 0.1
                5. I should study the difference between Adam and AdamW but it
                should be ok to use AdamW, if I know "why?"
                6. that is the main question - I believe the lr for target
                encoder should be really low as it is already finetuned and
                "just" needs to adapt so 1e-5 seems reasonable, but the
                discriminator is initialized randomly so it is classical
                finetuning settings where 2e-5 would make sense
                7. definitely - maybe apply bigger decay to solve the point 6
                8. 4 epochs
        Args:
            source train and val datasets - must be the same as they were for
            finetuning the selected classification head
            target dataset w/o labels - no subsetting, take the full dataset
            learning hyperparameters (optimizer, lr_schedules, temperature,
            loss combination parameters)
            metrics to track during the training
        Saves:
            the target encoder every epoch
            save also the Distcriminator? why not
            training info - train_loss, val_loss, val_metrics of the target
            encoder on the source task

        Get distilation labels before the start of the training = inference of
        the source encoder and classifier with temperature T on source train.
        """

        source_encoder = self.source_encoder
        target_encoder = self.target_encoder
        classifier = self.classifier
        discriminator = self.discriminator

        # save start time of the training
        start_time = time.strftime("%Y%m%d-%H%M%S")

        # set correct states for model parts
        source_encoder.eval()
        classifier.eval()
        target_encoder.train()
        discriminator.train()

        # setup criterion and optimizer
        bce_loss = torch.nn.BCELoss()
        kldiv_loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

        # get optimizer for each classfication head and the shared encoder
        if lr_decay is None or lr_decay == 1:
            encoder_optimizer = optimizer(encoder.parameters(), **optimizer_params)
            discriminator_optimizer = optimizer(
                discriminator.parameters(), **optimizer_params
            )
        else:
            list_of_layers = target_encoder.encoder.encoder.layer
            optimizer_params_list = layer_wise_learning_rate(
                list_of_layers,
                optimizer_params["lr"],
                lr_decay,
                len(discriminator.model),
            )
            encoder_optimizer = optimizer(optimizer_params_list, **optimizer_params)
            list_of_layers = list(discriminator.model)
            optimizer_params_list = layer_wise_learning_rate(
                list_of_layers,
                optimizer_params["lr"],
                lr_decay,
                0,
            )
            discriminator_optimizer = optimizer(
                optimizer_params_list, **optimizer_params
            )

        # get the total num of training steps to init lr_scheduler
        num_steps_per_epoch = len(target.torch_dataloader)
        num_training_steps = num_steps_per_epoch * num_epochs

        # get lr schedulers
        lr_schedulers = [
            lr_scheduler_call(
                optimizer=optim,
                num_warmup_steps=num_training_steps * warmup_steps_proportion,
                num_training_steps=num_training_steps,
            )
            for optim in [encoder_optimizer, discriminator_optimizer]
        ]

        # all models to device, gpu if available, else cpu
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        [
            model.to(device)
            for model in [source_encoder, target_encoder, classifier, discriminator]
        ]

        # track training info
        val_metrics = {metric: load(metric) for metric in metrics}
        val_metrics_progress = {metric_name: [] for metric_name in val_metrics}
        val_loss_mean_progress = []
        train_mean_loss_dict = {loss: [] for loss in ["disc", "gen", "class", "enc"]}
        train_batch_loss_dict = {loss: [] for loss in ["disc", "gen", "class", "enc"]}

        # prepare labels for distilation
        batch_size = source_train.torch_dataloader.batch_size
        num_workers = source_train.torch_dataloader.num_workers
        source_train.create_dataloader(
            batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        y_pred = torch.empty(0)
        logger.info("Getting the labels for knowledge distillation.")
        progress_bar = tqdm(range(len(source_train.torch_dataloader)))
        for batch in source_train.torch_dataloader:
            with torch.no_grad():
                source_features = source_encoder(**batch)
                probs = classifier(source_features)
                dist_probs = torch.sigmoid(inverted_sigmoid(probs) / temperature)
            y_pred = torch.cat((y_pred, dist_probs), 0)
            progress_bar.update(1)
        source_train.y = pd.Series(y_pred.flatten().numpy(), name="label")
        source_train.create_dataset()
        source_train.create_dataloader(
            batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        # display training progress
        progress_bar = tqdm(range(num_training_steps))
        counter = 0
        display_loss_after_iters = math.ceil(num_steps_per_epoch / 100)

        logger.info("Starting the adaptation")
        for epoch in range(num_epochs):
            # zip source and target data pairs
            dataloaders_zipped = zip(
                source_train.torch_dataloader, target.torch_dataloader
            )

            # lists for tracking the metrics
            train_loss_dict = {loss: [] for loss in ["disc", "gen", "class", "enc"]}
            train_epoch_loss_dict = {
                loss: [] for loss in ["disc", "gen", "class", "enc"]
            }

            # target encoder to train state
            target_encoder.train()

            for source_batch, target_batch in dataloaders_zipped:
                # tensors to cuda
                source_batch = {k: v.to(device) for k, v in source_batch.items()}
                target_batch = {k: v.to(device) for k, v in target_batch.items()}

                # zero gradients for optimizer
                discriminator_optimizer.zero_grad()

                # encoding the source sample by both encoders and target sample by
                # target encoder
                with torch.no_grad():
                    src_feat_src_enc = source_encoder(**source_batch)
                src_feat_tgt_enc = target_encoder(**source_batch)
                tgt_feat_tgt_enc = target_encoder(**target_batch)

                # concat src_src with tgt_tgt for discriminator training
                concat_feat = torch.cat((src_feat_src_enc, tgt_feat_tgt_enc), 0)

                # predict on discriminator
                domain_pred = discriminator(concat_feat.detach())

                # prepare real (discriminator) and fake (target encoder) label
                label_src = to_cuda(torch.ones(src_feat_src_enc.size(0))).unsqueeze(1)
                label_tgt = to_cuda(torch.zeros(tgt_feat_tgt_enc.size(0))).unsqueeze(1)
                domain = torch.cat((label_src, label_tgt), 0)

                # compute loss for discriminator - standard binary classification loss
                discriminator_loss = bce_loss(domain_pred, domain)
                discriminator_loss.backward()

                # update discriminator's weights
                discriminator_optimizer.step()

                # zero gradients for target encoder optimizer
                encoder_optimizer.zero_grad()

                # predict the domain of target domain encoded by the target encoder
                fake_domain_pred = discriminator(tgt_feat_tgt_enc)
                # compute loss like if it was source domain encoded
                gen_loss = bce_loss(fake_domain_pred, label_src)

                # logits for KL-divergence
                # for correct computation of KL-divergence I need to "complete"
                # the distibution - e.g. from prob 0.3 of class 0 make (0.3, 0.7)
                src_prob = source_batch["labels"].unsqueeze(1)
                log_src_prob = get_log_prob_for_kl_div(src_prob)
                tgt_prob = torch.sigmoid(
                    inverted_sigmoid(classifier(src_feat_tgt_enc)) / temperature
                )
                log_tgt_prob = get_log_prob_for_kl_div(tgt_prob)

                kd_loss = (
                    kldiv_loss(log_tgt_prob, log_src_prob.detach())
                    * temperature
                    * temperature
                )

                # compute the combined loss for target encoder
                encoder_loss = (
                    loss_combination_params[0] * gen_loss
                    + loss_combination_params[1] * kd_loss
                )
                encoder_loss.backward()

                # optimize target encoder
                encoder_optimizer.step()

                # update learning rates
                [lr_scheduler.step() for lr_scheduler in lr_schedulers]
                progress_bar.update(1)

                # save (and display training loss)
                loss_list = [discriminator_loss, gen_loss, kd_loss, encoder_loss]
                {
                    train_loss_dict[name].append(loss.item())
                    for name, loss in zip(train_loss_dict, loss_list)
                }
                {
                    train_epoch_loss_dict[name].append(loss.item())
                    for name, loss in zip(train_loss_dict, loss_list)
                }
                if counter % display_loss_after_iters == 0:
                    [
                        print(
                            "Training loss - ",
                            loss_name,
                            ": ",
                            np.mean(np.array(train_loss_dict[loss_name])),
                            sep="",
                        )
                        for loss_name in train_loss_dict
                    ]
                    train_loss_dict = {
                        loss: [] for loss in ["disc", "gen", "class", "enc"]
                    }

            # validation
            target_encoder.eval()
            val_epoch_loss_progress = []
            for batch in source_val.torch_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    features = target_encoder(**batch)
                    probs = classifier(features)
                cls_loss = bce_loss(
                    probs, batch["labels"].unsqueeze(1).to(torch.float32)
                )
                val_epoch_loss_progress.append(cls_loss.item())
                predictions = torch.round(probs)
                [
                    val_metrics[val_metric].add_batch(
                        predictions=predictions, references=batch["labels"]
                    )
                    for val_metric in val_metrics
                ]
            [
                val_metrics_progress[val_metric].append(
                    val_metrics[val_metric].compute()[val_metric]
                )
                for val_metric in val_metrics
            ]

            # compute average losses per epoch
            {
                train_batch_loss_dict[loss].append(train_epoch_loss_dict[loss])
                for loss in train_epoch_loss_dict
            }
            {
                train_mean_loss_dict[loss].append(
                    np.mean(np.array(train_epoch_loss_dict[loss]))
                )
                for loss in train_epoch_loss_dict
            }
            [
                print(
                    "Mean training loss - ",
                    loss_name,
                    ": ",
                    np.mean(np.array(train_mean_loss_dict[loss_name])),
                    sep="",
                )
                for loss_name in train_mean_loss_dict
            ]
            epoch_val_loss = np.mean(np.array(val_epoch_loss_progress))
            val_loss_mean_progress.append(epoch_val_loss)
            print(f"Mean validation loss for epoch {epoch}: {epoch_val_loss}")

            # save all models from the epoch
            # encoder
            save_path = os.path.join(
                paths.OUTPUT_MODELS_ADAPTED_ENCODER,
                "_".join([self.name, start_time, str(epoch)]),
            )
            os.makedirs(os.path.split(save_path)[0], exist_ok=True)
            torch.save(target_encoder.state_dict(), save_path)
            # discriminator
            save_path = os.path.join(
                paths.OUTPUT_MODELS_ADAPTED_DISCRIMINATOR,
                "_".join([self.name, start_time, str(epoch)]),
            )
            os.makedirs(os.path.split(save_path)[0], exist_ok=True)
            torch.save(discriminator.state_dict(), save_path)

        # save the training info
        val_metrics_progress["val_loss"] = val_loss_mean_progress
        train_mean_loss_dict_renamed = {
            "train_" + loss: train_mean_loss_dict[loss] for loss in train_mean_loss_dict
        }
        val_metrics_progress.update(train_mean_loss_dict_renamed)

        info_save_path = os.path.join(
            paths.OUTPUT_INFO_ADAPTATION,
            "_".join([self.name, "val", start_time]) + ".json",
        )
        os.makedirs(os.path.split(info_save_path)[0], exist_ok=True)
        with open(info_save_path, "w+") as fp:
            json.dump(val_metrics_progress, fp)

        info_save_path = os.path.join(
            paths.OUTPUT_INFO_ADAPTATION,
            "_".join([self.name, "train", start_time]) + ".json",
        )
        os.makedirs(os.path.split(info_save_path)[0], exist_ok=True)
        with open(info_save_path, "w+") as fp:
            json.dump(train_batch_loss_dict, fp)
        pass

    def predict(self, texts: List[str], predict_probs=True, temperature=1):
        self.target_encoder.eval()
        self.classifier.eval()
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        preprocessed_texts = self.preprocessor.preprocess(texts)
        X_tok = self.tokenizer.tokenize(texts)
        X_tok = {k: torch.tensor(v).to(device) for k, v in X_tok.items()}
        with torch.no_grad():
            features = self.target_encoder(**X_tok)
            pred = self.classifier(features)

        if temperature != 1:
            pred = torch.sigmoid(inverted_sigmoid(pred) / temperature)

        if not predict_probs:
            pred = torch.round(pred)

        pred = pred.flatten().tolist()
        return pred

    def bulk_predict(
        self, dataset: Type[ClassificationDataset], predict_probs=True, temperature=1
    ):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        preds = []
        for batch in dataset.torch_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                features = self.target_encoder(**batch)
                pred = self.classifier(features)
            if temperature != 1:
                pred = torch.sigmoid(inverted_sigmoid(pred) / temperature)
            if not predict_probs:
                pred = torch.round(pred)
            pred = pred.flatten().tolist()
            preds += pred
        dataset.y_pred = preds
        return preds


if __name__ == "__main__":

    from transformers import (
        ElectraTokenizerFast,
        ElectraModel,
        AutoModel,
        AutoModelForSequenceClassification,
        RobertaModel,
    )
    import torch

    tokenizer = ElectraTokenizerFast.from_pretrained("Seznam/small-e-czech")
    model = ElectraModel.from_pretrained("Seznam/small-e-czech")
    electra = AutoModel.from_pretrained("Seznam/small-e-czech")
    model = AutoModelForSequenceClassification.from_pretrained("Seznam/small-e-czech")
    model = AutoModelForSequenceClassification.from_pretrained("ufal/robeczech-base")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    roberta = AutoModel.from_pretrained("ufal/robeczech-base")
    roberta = RobertaModel.from_pretrained(
        "ufal/robeczech-base", output_hidden_states=True
    )
    bert = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()
    model = roberta

    dir(model)
    model.encoder.layer[0]

    tok = tokenizer(
        ["Daniel Štancl mi tímto modelem udělal velkou radost", "necum"],
        padding="max_length",
        max_length=512,
    )
    tok = {k: torch.tensor(tok[k]) for k in tok}
    out = model(**tok)
    out[0][:, 0, :].shape
    out[1].shape
    out[0][:, 0, :]
    out[1]
    out.keys()
    dir(out)
    out.hidden_states[12] == out.last_hidden_state
    out.last_hidden_state.shape
    out.last_hidden_state[:, 0]
    out.last_hidden_state
    out[0].shape
    out[0][:, 0]
    out[0][:, 0, :]
    dir(out)
    out.last_hidden_state.shape
    out.values()

    pass
