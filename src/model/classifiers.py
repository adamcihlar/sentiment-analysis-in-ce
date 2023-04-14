import itertools
import json
import math
import os
import random
import time
from typing import Callable, Dict, List, Type

import numpy as np
import pandas as pd
import torch
from coral_pytorch.dataset import corn_label_from_logits, levels_from_labelbatch
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import coral_loss, corn_loss
from evaluate import load
from loguru import logger
from sklearn.metrics import confusion_matrix
from hdbscan import HDBSCAN
from src.config import paths
from src.config.parameters import ClassifierParams, DiscriminatorParams
from src.model.encoders import Encoder
from src.model.tokenizers import Tokenizer
from src.utils.datasets import ClassificationDataset, save_train_info
from src.utils.optimization import (
    get_log_prob_for_kl_div,
    inverted_sigmoid,
    layer_wise_learning_rate,
    to_cuda,
)
from src.utils.text_preprocessing import Preprocessor
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import PCA


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
        num_classes=ClassifierParams.NUM_CLASSES,
        dropout=ClassifierParams.DROPOUT,
        model=None,
        path_to_finetuned=None,
        task_settings=ClassifierParams.TASK,
    ):
        """
        Based on the approach (ordinal/multiclass), the final layer is defined.
        """
        super(ClassificationHead, self).__init__()
        if task_settings == "ordinal":
            output_neurons = num_classes - 1
        elif task_settings == "multiclass":
            output_neurons = num_classes
        else:
            raise NotImplementedError
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
                    torch.nn.Linear(hidden_size, output_neurons),
                ),
            )
        if path_to_finetuned is not None:
            logger.info(
                f"Loading model parameters for ClassificationHead from {path_to_finetuned}."
            )
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            self.load_state_dict(
                torch.load(path_to_finetuned, map_location=torch.device(device))
            )
        self.num_classes = num_classes
        self.task_settings = task_settings
        self.first_layer = self.model[0]

    def forward(self, inputs, output_hidden=False):
        logits = self.model(inputs)
        if self.task_settings == "ordinal":
            probs = torch.sigmoid(logits)
        elif self.task_settings == "multiclass":
            probs = torch.softmax(logits, dim=1)
        if output_hidden:
            hidden = self.first_layer(inputs)
            return logits, probs, hidden
        return logits, probs

    def logits_to_scale(self, logits):
        cond_probs = torch.sigmoid(logits)
        probs = torch.cumprod(cond_probs, dim=1)
        scale = torch.mean(probs, dim=1)
        return scale


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
                    torch.nn.Tanh(),
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
        task_settings="ordinal",
        sim_dist=None,
        hiddens_norm=None,
        y_anchor=None,
        pca=None,
        hidden_layer=None,
    ):
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        if inference_mode:
            self.source_encoder = None
            self.name = target_encoder.name
            self.full_name = os.path.basename(classifier_checkpoint_path)
        else:
            self.source_encoder = source_encoder
            self.discriminator = discriminator
            self.name = self.source_encoder.name
        if classifier_checkpoint_path is not None:
            self.classifier = classifier(
                path_to_finetuned=classifier_checkpoint_path,
                task_settings=task_settings,
            )
        else:
            self.classifier = classifier
        self.target_encoder = target_encoder
        self.task_settings = task_settings
        self.sim_dist = sim_dist
        self.hiddens_norm = hiddens_norm
        self.pca = pca
        self.layer = hidden_layer
        self.dim_size = -1

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
        task: str,
        share_classifier: bool = False,
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
        classifiers = {
            ds_name: self.classifier(task_settings=task) for ds_name in train_datasets
        }
        encoder = self.source_encoder
        num_classes = self.classifier(task_settings=task).num_classes
        task = self.classifier(task_settings=task).task_settings

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
        if num_classes <= 2:
            val_metrics = {metric: load(metric) for metric in metrics}
        else:
            val_metrics = {
                metric: load("f1") for metric in ["micro", "macro", "weighted"]
            }
        val_metrics_progress = {ds: {} for ds in val_datasets}
        for ds in val_metrics_progress:
            if num_classes <= 2:
                val_metrics_progress[ds] = {
                    metric_name: [] for metric_name in val_metrics
                }
            else:
                val_metrics_progress[ds] = {
                    metric_name: [] for metric_name in ["micro", "macro", "weighted"]
                }
        train_loss_mean_progress = {ds_name: [] for ds_name in train_datasets}
        val_loss_mean_progress = {ds_name: [] for ds_name in val_datasets}
        train_loss_batch_progress = {ds_name: [] for ds_name in train_datasets}

        # display training progress
        progress_bar = tqdm(range(num_training_steps))
        counter = 1
        display_loss_after_iters = math.ceil(
            sum(num_steps_per_epoch_per_dataloader.values()) / 100
        )

        # loss for multiclass classification
        ce_loss = torch.nn.CrossEntropyLoss()

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
                # levels = levels_from_labelbatch(batch["labels"], num_classes).to(device)
                # forward pass
                features = encoder(**batch)
                logits, probs = classifiers[source_ds].forward(features)
                # backward pass
                if task == "ordinal":
                    cls_loss = corn_loss(logits, batch["labels"], num_classes)
                elif task == "multiclass":
                    cls_loss = ce_loss(logits, batch["labels"])
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
                        if share_classifier:
                            logits, probs = list(classifiers.values())[0].forward(
                                features
                            )
                        else:
                            logits, probs = classifiers[val_ds_name].forward(features)

                    if task == "ordinal":
                        cls_loss = corn_loss(logits, batch["labels"], num_classes)
                        predictions = corn_label_from_logits(logits).float()
                    elif task == "multiclass":
                        cls_loss = ce_loss(logits, batch["labels"])
                        predictions = torch.argmax(probs, dim=1)
                    val_epoch_loss_progress[val_ds_name].append(cls_loss.item())
                    [
                        val_metrics[val_metric].add_batch(
                            predictions=predictions, references=batch["labels"]
                        )
                        for val_metric in val_metrics
                    ]
                if num_classes <= 2:
                    [
                        val_metrics_progress[val_ds_name][val_metric].append(
                            val_metrics[val_metric].compute()[val_metric]
                        )
                        for val_metric in val_metrics
                    ]
                else:
                    [
                        val_metrics_progress[val_ds_name][val_metric].append(
                            val_metrics[val_metric].compute(average=val_metric)["f1"]
                        )
                        for val_metric in list(val_metrics_progress[val_ds_name].keys())
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
            self.save_model(encoder, save_path)

            # classifiers
            cls_save_paths = [
                os.path.join(
                    paths.OUTPUT_MODELS_FINETUNED_CLASSIFIER,
                    "_".join([self.name, start_time, cls, str(epoch)]),
                )
                for cls in classifiers
            ]
            [
                self.save_model(classifiers[cls_name], path)
                for cls_name, path in zip(classifiers, cls_save_paths)
            ]

        # save the training info
        for ds_name in val_datasets:
            val_metrics_progress[ds_name]["val_loss"] = val_loss_mean_progress[ds_name]
            if share_classifier:
                val_metrics_progress[ds_name]["train_loss"] = list(
                    train_loss_mean_progress.values()
                )[0]
            else:
                val_metrics_progress[ds_name]["train_loss"] = train_loss_mean_progress[
                    ds_name
                ]
        info_save_path = os.path.join(
            paths.OUTPUT_INFO_FINETUNING,
            "_".join([self.name, "val", start_time]) + ".json",
        )
        save_train_info(val_metrics_progress, info_save_path)

        info_save_path = os.path.join(
            paths.OUTPUT_INFO_FINETUNING,
            "_".join([self.name, "train", start_time]) + ".json",
        )
        save_train_info(train_loss_batch_progress, info_save_path)

        # save datasets as I will need them for adapt method
        for ds_name in train_datasets:
            save_path = os.path.join(
                paths.DATA_FINAL_SOURCE_TRAIN,
                "_".join([self.name, start_time, ds_name]) + ".csv",
            )
            train_datasets[ds_name].save_data(save_path)
        for ds_name in val_datasets:
            save_path = os.path.join(
                paths.DATA_FINAL_SOURCE_VAL,
                "_".join([self.name, start_time, ds_name]) + ".csv",
            )
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
        target_name="",
        source_model="",
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
        val_metrics = {metric: load("f1") for metric in ["micro", "macro", "weighted"]}
        val_metrics_progress = {
            metric_name: [] for metric_name in ["micro", "macro", "weighted"]
        }
        val_loss_mean_progress = []
        test_metrics = {metric: load("f1") for metric in ["micro", "macro", "weighted"]}
        test_metrics_progress = {
            metric_name: [] for metric_name in ["micro", "macro", "weighted"]
        }
        test_loss_mean_progress = []
        train_mean_loss_dict = {loss: [] for loss in ["disc", "gen", "class", "enc"]}
        train_batch_loss_dict = {loss: [] for loss in ["disc", "gen", "class", "enc"]}

        # get the initial metrics for validation dataset
        logger.info("Getting the initial metrics for validation dataset.")
        target_encoder.eval()
        val_epoch_loss_progress = []
        progress_bar = tqdm(range(len(source_val.torch_dataloader)))
        for batch in source_val.torch_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                features = target_encoder(**batch)
                logits, probs = classifier(features)
                cls_loss = corn_loss(logits, batch["labels"], classifier.num_classes)
                predictions = corn_label_from_logits(logits).float()
                progress_bar.update(1)
                val_epoch_loss_progress.append(cls_loss.item())
                [
                    val_metrics[val_metric].add_batch(
                        predictions=predictions, references=batch["labels"]
                    )
                    for val_metric in val_metrics
                ]
                [
                    val_metrics_progress[val_metric].append(
                        val_metrics[val_metric].compute(average=val_metric)["f1"]
                    )
                    for val_metric in list(val_metrics_progress.keys())
                ]
                # print validation info
                epoch_val_loss = np.mean(np.array(val_epoch_loss_progress))
                val_loss_mean_progress.append(epoch_val_loss)
                print(f"Mean validation loss of the original encoder {epoch_val_loss}")

        # get initial test metrics if target labels are available
        if target.y is not None:
            logger.info("Getting the initial metrics for target dataset.")
            progress_bar = tqdm(range(len(target.torch_dataloader)))
            y_pred = []
            test_epoch_loss_progress = []
            for batch in target.torch_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    features = target_encoder(**batch)
                    logits, probs = classifier(features)
                cls_loss = corn_loss(logits, batch["labels"], classifier.num_classes)
                predictions = corn_label_from_logits(logits).float()
                y_pred += predictions.detach().cpu().tolist()
                progress_bar.update(1)
                test_epoch_loss_progress.append(cls_loss.item())
                [
                    test_metrics[test_metric].add_batch(
                        predictions=predictions, references=batch["labels"]
                    )
                    for test_metric in test_metrics
                ]
            [
                test_metrics_progress[test_metric].append(
                    test_metrics[test_metric].compute(average=test_metric)["f1"]
                )
                for test_metric in list(test_metrics_progress.keys())
            ]
            # print test info
            print(confusion_matrix(target.y, y_pred))
            epoch_test_loss = np.mean(np.array(test_epoch_loss_progress))
            test_loss_mean_progress.append(epoch_test_loss)
            print(f"Mean test loss of the original encoder {epoch_test_loss}")

        # prepare labels for distilation
        batch_size = source_train.torch_dataloader.batch_size
        num_workers = source_train.torch_dataloader.num_workers
        source_train.create_dataloader(
            batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        y_pred = torch.empty(0).to(device)
        logger.info("Getting the labels for knowledge distillation.")
        progress_bar = tqdm(range(len(source_train.torch_dataloader)))
        for batch in source_train.torch_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                source_features = source_encoder(**batch)
                logits, probs = classifier(source_features)
                dist_probs = torch.sigmoid(logits / temperature)
            y_pred = torch.cat((y_pred, dist_probs), 0)
            progress_bar.update(1)
        # source_train.y = pd.Series(y_pred.numpy(), name="label")
        source_train.y = list(y_pred.cpu().numpy())
        source_train.create_dataset()
        source_train.create_dataloader(
            batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        # display training progress
        progress_bar = tqdm(range(num_training_steps * 2))
        counter = 0
        display_loss_after_iters = math.ceil(num_steps_per_epoch / 10)

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

            # target encoder to eval state and discriminator to train state for
            # discriminator training
            target_encoder.eval()
            discriminator.train()

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

                # update discriminator's weights and learning rate
                discriminator_optimizer.step()
                lr_schedulers[0].step()

                progress_bar.update(1)
                counter += 1

                # save (and display training loss)
                train_loss_dict["disc"].append(discriminator_loss.item())
                train_epoch_loss_dict["disc"].append(discriminator_loss.item())
                if counter % display_loss_after_iters == 0:
                    print(
                        "Training loss - disc: ",
                        np.mean(np.array(train_loss_dict["disc"])),
                        sep="",
                    )
                    train_loss_dict["disc"] = []

            # rebuild dataloaders
            dataloaders_zipped = zip(
                source_train.torch_dataloader, target.torch_dataloader
            )

            # discriminator to eval state and target encoder to train state for
            # encoder training
            target_encoder.train()
            discriminator.eval()

            for source_batch, target_batch in dataloaders_zipped:
                # tensors to cuda
                source_batch = {k: v.to(device) for k, v in source_batch.items()}
                target_batch = {k: v.to(device) for k, v in target_batch.items()}

                src_feat_tgt_enc = target_encoder(**source_batch)
                tgt_feat_tgt_enc = target_encoder(**target_batch)
                # zero gradients for target encoder optimizer
                encoder_optimizer.zero_grad()

                # predict the domain of target domain encoded by the target encoder
                fake_domain_pred = discriminator(tgt_feat_tgt_enc)
                # compute loss like if it was source domain encoded
                label_src = to_cuda(torch.ones(tgt_feat_tgt_enc.size(0))).unsqueeze(1)
                gen_loss = bce_loss(fake_domain_pred, label_src)

                # logits for KL-divergence
                # for correct computation of KL-divergence I need to "complete"
                # the distibution - e.g. from prob 0.3 of class 0 make (0.3, 0.7)
                src_prob = source_batch["labels"]
                log_src_prob = get_log_prob_for_kl_div(src_prob)
                tgt_prob = torch.sigmoid(classifier(src_feat_tgt_enc)[0] / temperature)
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

                # update encoder's learning rates
                lr_schedulers[1].step()

                progress_bar.update(1)
                counter += 1

                # save (and display training loss)
                loss_list = [gen_loss, kd_loss, encoder_loss]
                {
                    train_loss_dict[name].append(loss.item())
                    for name, loss in zip(list(train_loss_dict)[1:], loss_list)
                }
                {
                    train_epoch_loss_dict[name].append(loss.item())
                    for name, loss in zip(list(train_loss_dict)[1:], loss_list)
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
                        for loss_name in list(train_loss_dict)[1:]
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
                    logits, probs = classifier(features)
                cls_loss = corn_loss(logits, batch["labels"], classifier.num_classes)
                predictions = corn_label_from_logits(logits).float()
                val_epoch_loss_progress.append(cls_loss.item())
                [
                    val_metrics[val_metric].add_batch(
                        predictions=predictions, references=batch["labels"]
                    )
                    for val_metric in val_metrics
                ]
            [
                val_metrics_progress[val_metric].append(
                    val_metrics[val_metric].compute(average=val_metric)["f1"]
                )
                for val_metric in list(val_metrics_progress.keys())
            ]

            # print validation info
            epoch_val_loss = np.mean(np.array(val_epoch_loss_progress))
            val_loss_mean_progress.append(epoch_val_loss)
            print(f"Mean validation loss for epoch {epoch}: {epoch_val_loss}")

            #             # test if target labels are available
            #             if target.y is not None:
            #                 y_pred = []
            #                 test_epoch_loss_progress = []
            #                 for batch in target.torch_dataloader:
            #                     batch = {k: v.to(device) for k, v in batch.items()}
            #                     with torch.no_grad():
            #                         features = target_encoder(**batch)
            #                         logits, probs = classifier(features)
            #                     cls_loss = corn_loss(
            #                         logits, batch["labels"], classifier.num_classes
            #                     )
            #                     predictions = corn_label_from_logits(logits).float()
            #                     y_pred += predictions.detach().cpu().tolist()
            #                     test_epoch_loss_progress.append(cls_loss.item())
            #                     [
            #                         test_metrics[test_metric].add_batch(
            #                             predictions=predictions, references=batch["labels"]
            #                         )
            #                         for test_metric in test_metrics
            #                     ]
            #                 [
            #                     test_metrics_progress[test_metric].append(
            #                         test_metrics[test_metric].compute(average=test_metric)["f1"]
            #                     )
            #                     for test_metric in list(test_metrics_progress.keys())
            #                 ]
            #                 print(confusion_matrix(target.y, y_pred))
            #                 epoch_test_loss = np.mean(np.array(test_epoch_loss_progress))
            #                 test_loss_mean_progress.append(epoch_test_loss)
            #                 print(f"Mean test loss on the target data {epoch_test_loss}")

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

        # save the final  model
        # encoder
        # TODO save with better name
        save_path = os.path.join(
            paths.OUTPUT_MODELS_ADAPTED_ENCODER,
            "_".join(
                [self.target_name, self.name, start_time, str(epoch), self.source_model]
            ),
        )
        self.save_model(target_encoder, save_path)
        # discriminator
        save_path = os.path.join(
            paths.OUTPUT_MODELS_ADAPTED_DISCRIMINATOR,
            "_".join([self.name, start_time, str(epoch)]),
        )
        self.save_model(discriminator, save_path)

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
        save_train_info(val_metrics_progress, info_save_path)

        test_metrics_progress["test_loss"] = test_loss_mean_progress
        info_save_path = os.path.join(
            paths.OUTPUT_INFO_ADAPTATION,
            "_".join([self.name, "test", start_time]) + ".json",
        )
        save_train_info(test_metrics_progress, info_save_path)

        info_save_path = os.path.join(
            paths.OUTPUT_INFO_ADAPTATION,
            "_".join([self.name, "train", start_time]) + ".json",
        )
        save_train_info(train_batch_loss_dict, info_save_path)

        pass

    def predict(
        self,
        texts: List[str],
        predict_scale=True,
        temperature=1,
        output_hidden=False,
    ):
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
            if output_hidden == -1:
                logits, probs, hidden = self.classifier(features, output_hidden)
            elif output_hidden == -2:
                hidden = features
                logits, probs = self.classifier(features)
            else:
                logits, probs = self.classifier(features)

        if temperature != 1:
            logits = logits / temperature

        if self.task_settings == "ordinal":
            if predict_scale:
                pred = self.classifier.logits_to_scale(logits)
            else:
                pred = corn_label_from_logits(logits).float()
        elif self.task_settings == "multiclass":
            pred = torch.argmax(logits, dim=1).float()
        pred = pred.flatten().tolist()
        if output_hidden:
            return pred, hidden
        return pred

    def bulk_predict(
        self,
        dataset: Type[ClassificationDataset],
        predict_scale=True,
        temperature=1,
        output_hidden=False,
    ):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.target_encoder.eval()
        self.classifier.eval()
        preds = []
        hiddens = []
        n_steps = len(dataset.torch_dataloader)
        progress_bar = tqdm(range(n_steps))
        for batch in dataset.torch_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                features = self.target_encoder(**batch)
                if output_hidden == -1:
                    logits, probs, hidden = self.classifier(features, output_hidden)
                elif output_hidden == -2:
                    hidden = features
                    logits, probs = self.classifier(features)
                else:
                    logits, probs = self.classifier(features)

            if temperature != 1:
                logits = logits / temperature

            if self.task_settings == "ordinal":
                if predict_scale:
                    pred = self.classifier.logits_to_scale(logits)
                else:
                    pred = corn_label_from_logits(logits).float()
            elif self.task_settings == "multiclass":
                pred = torch.argmax(logits, dim=1).float()
            pred = pred.flatten().tolist()
            preds += pred

            if output_hidden:
                hiddens.append(hidden)

            progress_bar.update(1)
        dataset.y_pred = preds

        if output_hidden:
            hiddens = torch.cat(hiddens, dim=0)
            return preds, hiddens
        return preds

    def get_nn_sim_distribution(self, target_ds, nn=True, layer=-1, dim_size=None):
        """
        Get distribution of nearest neighbors similarities of the target
        dataset.
        Return vector of cosine similarities and save it as attribute.
        """
        if self.hiddens_norm is None:
            preds, hiddens = self.bulk_predict(
                target_ds, predict_scale=True, output_hidden=layer
            )
            self.layer = layer

            if dim_size:
                self.pca = PCA(dim_size)
                hiddens = torch.tensor(self.pca.fit_transform(hiddens.detach().numpy()))

            hiddens_norm = torch.nn.functional.normalize(hiddens, dim=1)
            self.hiddens_norm = hiddens_norm

        # cosine similarity
        cos_sim_mat = torch.mm(hiddens_norm, hiddens_norm.transpose(0, 1))
        cos_sim_mat.fill_diagonal_(-2)

        # nearest neighbor
        if nn:
            sim_dist = cos_sim_mat.max(dim=0).values
        else:
            sim_dist = cos_sim_mat[cos_sim_mat != torch.tensor(-2)]
        self.sim_dist = sim_dist
        # TODO labels divided by 2 to get [0,1] scale
        self.y_anchor = target_ds.y.loc[~target_ds.y.isna()] / 2
        self.anchor_hidden = torch.tensor(
            np.array(self.hiddens_norm)[~target_ds.y.isna()]
        )
        return sim_dist, hiddens_norm

    def nn_bulk_predict(self, target_ds, nn=True, layer=-1, dim_size=None):
        """
        Predict label based on the nearest neighbor.
        Returns label and confidence that is based on the empirical p-value of
        how likely it is that the test sample is the nearest neighbor of the anchor sample
        the similarity and similarities distribution.
        """

        self.get_nn_sim_distribution(target_ds, nn=nn, layer=layer, dim_size=dim_size)
        test_hidden = torch.tensor(np.array(self.hiddens_norm)[target_ds.y.isna()])
        anchor_hidden = self.anchor_hidden

        cos_sim_anch_mat = torch.mm(test_hidden, anchor_hidden.transpose(0, 1))
        nn_ind = cos_sim_anch_mat.argmax(dim=1)
        y_pred_nn = self.y_anchor.iloc[nn_ind]

        sim_dist_anch = cos_sim_anch_mat.max(dim=1).values
        y_conf_nn = np.array(
            [
                (sum(val > self.sim_dist) / len(self.sim_dist)).detach()
                for val in sim_dist_anch
            ]
        )

        return y_pred_nn, y_conf_nn

    def mix_bulk_predict(self, target_ds, layer=None, dim_size=-1, scale=True):
        """
        Ensemble of nearest neighbor and classifier prediction.

        target_ds should contain some labeled samples (~anchor set)
        Based on distances in the target ds we assign weight to the nearest
        neighbor classifier and the prediction of the neural net.

        Returns series of predictions, they are also saved to the target_ds.
        """
        # nn prediction
        # use the values obtained during anchor set definition if available
        if layer is None:
            if self.layer is not None:
                layer = self.layer
            else:
                logger.error("Provide index of layer to get the embeddings from.")

        if dim_size == -1:
            dim_size = self.dim_size
        else:
            if 0 < dim_size < 1:
                dim_size = self._get_pca_dim_from_variance_ratio(hiddens, dim_size)
            self.dim_size = dim_size

        y_pred_nn, y_conf_nn = self.nn_bulk_predict(
            target_ds, layer=layer, dim_size=dim_size
        )
        y_pred_nn_w = y_pred_nn * y_conf_nn

        # cls prediction
        cls_conf = 1 - y_conf_nn
        y_pred_cls = np.array(target_ds.y_pred)[list(target_ds.y.isna())]
        y_pred_cls_w = y_pred_cls * cls_conf

        # combined prediction
        y_pred = y_pred_nn_w + y_pred_cls_w

        # save to target_ds
        target_ds.y_pred = pd.Series(target_ds.y_pred, index=target_ds.y.index)
        if scale:
            target_ds.y_pred.loc[~target_ds.y.isna()] = (
                target_ds.y.loc[~target_ds.y.isna()] / 2
            )
            target_ds.y_pred.loc[target_ds.y.isna()] = list(y_pred)
        else:
            target_ds.y_pred.loc[~target_ds.y.isna()] = target_ds.y.loc[
                ~target_ds.y.isna()
            ]
            target_ds.y_pred.loc[target_ds.y.isna()] = np.floor(y_pred * 3)

        return target_ds.y_pred

    def nn_predict(
        self,
        texts: List[str],
    ):
        assert (
            self.sim_dist is not None
        ), "Distribution of the nearest neighbors similarities not available. Call self.get_nn_sim_distribution(target_ds) first."
        preds, hiddens = self.predict(
            texts, predict_scale=True, output_hidden=self.layer
        )

        if self.pca is not None:
            hiddens = torch.tensor(self.pca.fit_transform(hiddens.detach().numpy()))

        # cosine similarity
        hiddens_norm = torch.nn.functional.normalize(hiddens, dim=1)
        cos_sim_anch_mat = torch.mm(hiddens_norm, self.anchor_hidden.transpose(0, 1))
        nn_ind = cos_sim_anch_mat.argmax(dim=1)
        y_pred_nn = self.y_anchor.iloc[nn_ind]

        sim_dist_anch = cos_sim_anch_mat.max(dim=1).values
        y_conf_nn = np.array(
            [
                (sum(val > self.sim_dist) / len(self.sim_dist)).detach()
                for val in sim_dist_anch
            ]
        )

        return y_pred_nn, y_conf_nn, preds

    def mix_predict(
        self,
        texts: List[str],
        scale=True,
    ):
        y_pred_nn, y_conf_nn, y_pred_cls = self.nn_predict(texts)
        cls_conf = 1 - y_conf_nn
        y_pred_nn_w = y_pred_nn * y_conf_nn
        y_pred_cls_w = y_pred_cls * cls_conf

        # combined prediction
        y_pred = y_pred_nn_w + y_pred_cls_w

        if not scale:
            y_pred = np.floor(y_pred * 3)
        return list(y_pred)

    def suggest_anchor_set(self, target_ds, layer=-1, dim_size=None):
        """
        Cluster the embeddings to suggest the most useful samples to label.

        Forward pass of the target_ds - I should save the normalized hidden state,
        to avoid computing is again in get_nn_sim_distribution. Save also the
        layer and dim_size, it seems reasonable to use the same one.

        Cluster - what algo? Set clusters count? The goal is to find very dense
        clusters that I want to influence with the nn prediction, I do not want
        to change max number of predictions but max number with high
        confidence - HDBSCAN. What hyperparameters
        Then centroids of the clusters.
        Samples closest to the centroids are the samples to label.
        """
        self.layer = layer
        preds, hiddens = self.bulk_predict(
            target_ds, predict_scale=True, output_hidden=layer
        )

        # if dim_size is ratio, compute the corresponding dim_size
        if 0 < dim_size < 1:
            dim_size = self._get_pca_dim_from_variance_ratio(hiddens, dim_size)
        self.dim_size = dim_size

        if dim_size:
            self.pca = PCA(dim_size)
            hiddens = torch.tensor(self.pca.fit_transform(hiddens.detach().numpy()))

        hiddens_norm = torch.nn.functional.normalize(hiddens, dim=1)
        self.hiddens_norm = hiddens_norm

        hiddens_norm_df = pd.DataFrame(hiddens_norm, index=target_ds.X.index)
        hiddens_norm_df = pd.concat(
            [
                hiddens_norm_df,
                pd.DataFrame({"cls": list(labs)}, index=target_ds.X.index),
            ],
            axis=1,
        )
        hiddens_norm_df = pd.concat([hiddens_norm_df, target_ds.y], axis=1)

        target_ds.y

        hdbscan = HDBSCAN(min_cluster_size=5)
        labs = hdbscan.fit_predict(hiddens_norm)
        labs_list = pd.Series(labs).unique()
        labs_list = labs_list.loc[labs_list != -1]

        for lab in labs_list:
            hiddens_norm_subs = hiddens_norm[labs == lab]
            center = np.mean(hiddens_norm_subs, dim=0)
            center_norm = torch.nn.functional.normalize(center_norm, dim=1)

            cos_sim_mat = torch.mm(hiddens_norm_subs, center_norm.transpose(0, 1))
            cos_sim_mat.fill_diagonal_(-2)

        pass

    def _get_pca_dim_from_variance_ratio(self, hiddens, ratio):
        pca = PCA(min(hiddens.shape))
        pca.fit(hiddens)
        dim = sum(np.cumsum(pca.explained_variance_ratio_) < ratio) + 1
        return dim

    def save_model(self, model, path):
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        saved = 0
        while saved != 1:
            try:
                torch.save(model.state_dict(), path)
            except IOError:
                pass
            else:
                saved = 1
                logger.info(f"Model saved at {path}.")
        pass


if __name__ == "__main__":

    import plotly.express as px

    fig = px.scatter(hiddens_norm_df, x=0, y=1, color="label")
    fig.write_html("output/assets/csfd_target_label.html")

    pass
