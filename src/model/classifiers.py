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
import torch
from transformers import (
    RobertaTokenizer,
    RobertaModel,
)
from torch.utils.data import DataLoader
from evaluate import load

from src.utils.datasets import Dataset
from src.utils.optimization import layer_wise_learning_rate

# from src.utils.custom_layers import Linear
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
    Layers wrapped to Sequential modules to enable iterating over them.
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
                torch.nn.Sequential(
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden_size, num_classes),
                    torch.nn.Sigmoid(),
                )
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
        self, input_size=768, hidden_size=3072, num_classes=1, dropout=0.0, model=None
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
        classifier_checkpoint_path,
    ):
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.source_encoder = source_encoder
        if classifier_checkpoint_path is not None:
            self.classifier = classifier(path_to_finetuned=classifier_checkpoint_path)
        else:
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
        source_train_dataset,
        source_val_dataset,
        target_dataset,
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
        source_train_dataset
        source_val_dataset
        target_dataset

        optimizer = AdamW
        optimizer_params = {"lr": 2e-5, "betas": (0.9, 0.999)}
        lr_decay = 0.9
        lr_scheduler_call = get_linear_schedule_with_warmup
        warmup_steps_proportion = 0.1
        num_epochs = 4
        metrics = ["f1", "accuracy", "precision", "recall"]

        source_encoder = asc.source_encoder
        target_encoder = asc.target_encoder
        classifier = asc.classifier(
            path_to_finetuned=paths.OUTPUT_MODELS_FINETUNED_CLASSIFIER_FINAL
        )
        discriminator = asc.discriminator

        # set correct states for model parts
        source_encoder.eval()
        classifier.eval()
        target_encoder.train()
        discriminator.train()

        # setup criterion and optimizer
        bce_loss = torch.nn.BCELoss()
        kldiv_loss = torch.nn.KLDivLoss(reduction="batchmean")

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

        ##############################################################################

        # TODO continue here with the adaptation method
        # decide how to pass the training details:
        # optimizer DONE
        # learning rates DONE
        # lr schedules
        # how to pass the same train and val source dataset?
        # basically how the finetune and adapt methods will be connected
        len_data_loader = min(len(src_data_loader), len(tgt_data_train_loader))

        for epoch in range(args.num_epochs):
            # zip source and target data pair
            data_zip = enumerate(zip(src_data_loader, tgt_data_train_loader))
            for step, (
                (reviews_src, src_mask, _),
                (reviews_tgt, tgt_mask, _),
            ) in data_zip:

                # for every step take samples from both domains
                reviews_src = make_cuda(reviews_src)
                src_mask = make_cuda(src_mask)
                reviews_tgt = make_cuda(reviews_tgt)
                tgt_mask = make_cuda(tgt_mask)

                # zero gradients for optimizer
                optimizer_D.zero_grad()

                # encoding of source sample by both encoders and target sample by
                # target encoder
                with torch.no_grad():
                    feat_src = src_encoder(reviews_src, src_mask)
                feat_src_tgt = tgt_encoder(reviews_src, src_mask)
                feat_tgt = tgt_encoder(reviews_tgt, tgt_mask)

                # feat_src = torch.zeros(1, 2, 768)
                # feat_src_tgt = torch.ones(1, 2, 768)
                # feat_tgt = torch.ones(1, 2, 768) + 1

                # concatenate source and target domain batches(!) encoded by
                # target (sic!) encoder
                # this is probably mistake in this implementation, because the
                # discriminator should see the source domain encoded by the source
                # encoder so that the target encoder can be getting as close as
                # possible to these "ground truth" source-source encodings
                # this is for the task for discriminator - it will try to
                # distinguish between source and target domains encoded by the same
                # (target (sic!)) encoder
                feat_concat = torch.cat((feat_src_tgt, feat_tgt), 0)

                # predict on discriminator
                pred_concat = discriminator(feat_concat.detach())

                # prepare real and fake label
                # why real and fake? So source domain = 1 and target domain = 0
                # this is what we show to the discriminator and it really corresponds to
                # the input domain.
                # However, while training the target encoder we optimize it to
                # encode target domain in such way, that it is classified by the
                # discriminator as source domain (=> "encoder learns to trick the
                # discriminator") - in that moment the labels are "fake" because
                # the sample comes from the target domain but we give it the label
                # of source domain
                label_src = make_cuda(torch.ones(feat_src_tgt.size(0))).unsqueeze(1)
                label_tgt = make_cuda(torch.zeros(feat_tgt.size(0))).unsqueeze(1)
                label_concat = torch.cat((label_src, label_tgt), 0)

                # compute loss for discriminator - standard binary classification loss
                dis_loss = BCELoss(pred_concat, label_concat)
                dis_loss.backward()

                # this sets all params of the discriminator to (by default) range [-0.01, 0.01]
                # I am not sure if I noticed this in the paper
                # yes, this is to make the training more stable
                for p in discriminator.parameters():
                    p.data.clamp_(-args.clip_value, args.clip_value)
                # optimize discriminator - updates only the discriminator's weights
                optimizer_D.step()

                # this is something I don't really undestand - there is only one
                # output neuron with sigmoid activation (typical binary
                # classification settings)
                # so the argmax below always returns only zeros as the shape of
                # pred_concat is always (batch_size, 1) so the max element has
                # always index 0
                # but it doesn't affect the optimization, it is just info about the
                # training
                pred_cls = torch.squeeze(pred_concat.max(1)[1])
                acc = (pred_cls == label_concat).float().mean()

                # zero gradients for target encoder optimizer
                optimizer_G.zero_grad()
                T = args.temperature

                # predict on discriminator
                # predict the domain of target domain encoded by the target encoder
                pred_tgt = discriminator(feat_tgt)
                # we want the target encoder to encode its target domain samples
                # like if it was a source domain sample - map it to the same space
                # that is why the loss is computed with respect to the source
                # domain label even though we encoded target domain sample
                gen_loss = BCELoss(pred_tgt, label_src)

                # logits for KL-divergence
                # classify the source encoded by both source and target encoder and
                # distill the knowledge from source encoder to target encoder
                # this is why it's once log(prob) and once just probability -
                # https://stackoverflow.com/questions/62806681/pytorch-kldivloss-loss-is-negative
                with torch.no_grad():
                    src_prob = F.softmax(src_classifier(feat_src) / T, dim=-1)
                tgt_prob = F.log_softmax(src_classifier(feat_src_tgt) / T, dim=-1)
                kd_loss = KLDivLoss(tgt_prob, src_prob.detach()) * T * T

                # compute the combined loss for target encoder
                loss_tgt = args.alpha * gen_loss + args.beta * kd_loss
                loss_tgt.backward()
                torch.nn.utils.clip_grad_norm_(
                    tgt_encoder.parameters(), args.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(tgt_encoder.parameters(), 1)
                # optimize target encoder
                optimizer_G.step()


if __name__ == "__main__":
    pass
