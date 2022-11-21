from typing import List, Type, Callable
import torch
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    RobertaForSequenceClassification,
)
from torch.utils.data import DataLoader

from src.utils.datasets import Dataset
from src.utils.optimization import layer_wise_learning_rate
from src.model.encoders import Encoder
from src.utils.text_preprocessing import Preprocessor
from src.model.tokenizers import Tokenizer


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

    def finetune(
        train_datasets: Dict,
        val_datasets: Dict,
        optimizer: Type[torch.optim.Optimizer],
        optimizer_params: Dict,
        lr_scheduler: Callable,
        num_epochs: int,
    ):
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
    num_epochs = 4

    # method body
    classifiers = {ds_name: asc.classifier() for ds_name in train_datasets}
    encoder = source_encoder

    cls_optimizers = {
        cls_name: optimizer(classifiers[cls_name].parameters(), **optimizer_params)
        for cls_name in classifiers
    }

    if layer_wise_lr_decay is None or layer_wise_lr_decay == 1:
        encoder_optimizer = optimizer(encoder.parameters(), **optimizer_params)
    else:
        list_of_layers = source_encoder.encoder.encoder.layer
        optimizer_params_list = layer_wise_learning_rate(
            list_of_layers, layer_wise_lr_decay
        )
        encoder_optimizer = optimizer(optimizer_params_list, **optimizer_params)

    len(train_datasets["facebook"].torch_dataloader)

    # specify training details
    optimizer = AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.999))
    num_epochs = 4
    num_training_steps = num_epochs * len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_training_steps * 0.1,
        num_training_steps=num_training_steps,
    )
    parameter_ids = [
        [id(p) for p in group["params"]] for group in optimizer_params_list
    ]
    parameter_ids[0]
    parameter_ids[3]
    source_encoder.encoder.param_groups
    dir(source_encoder)
