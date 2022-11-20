from typing import List
import torch
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    RobertaForSequenceClassification,
)
from torch.utils.data import DataLoader

from src.utils.datasets import Dataset


class ClassificationHead(torch.nn.Module):
        """
        Sentiment classifier with standard architecture.
        The output is probabilities, so it is needed to use BCELoss as loss
        function.
        If this was numerically unstable try skipping the sigmoid activation
        function and use BCEWithLogitsLoss instead.
        """

    def __init__(
        self, input_size=768, hidden_size=768, num_classes=1, dropout=0.1, model=None
    ):
        super(ClassificationHead, self).__init__()
        if model is not None:
            self.model = model
        else:
            self.model = torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                # torch.nn.Linear(input_size, hidden_size),
                # torch.nn.ReLU()
                torch.nn.Dropout(dropout)
                torch.nn.Linear(hidden_size, num_classes)
                torch.nn.Sigmoid()
            )

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs

class Discriminator(torch.nn.module):
    '''
    Classifier trained to distinguish between source and target domain.
    '''
    def __init__(
        self, input_size=768, hidden_size=3072, num_classes=1, dropout=0.0, model=None
    ):
        super(Discriminator, self).__init__()
        if model is not None:
            self.model = model
        else:
            self.model = torch.nn.Sequential(
                torch.nn.Linear(input_size, hidden_size),
                torch.nn.ReLU()
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU()
                torch.nn.Linear(hidden_size, num_classes)
                torch.nn.Sigmoid()
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
        classifier: List,
        discriminator,
        target_encoder=None,
    ):
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.source_encoder = source_encoder
        self.classifiers = classifiers
        self.target_encoder = target_encoder
        self.discriminator = discriminator

    def finetune(
        train_classification_dataset,
        val_classification_dataset,
        optimizer,
        lr_scheduler,
        num_epochs,
    ):
        # init the whole classification model
        # - concat the source_encoder with the classifier
        pass



if __name__ == "__main__":
    from src.model.encoders import Encoder
    encoder = Encoder()
    encoder
    source_encoder = RobertaModel.from_pretrained("ufal/robeczech-base")


    tok = tokenizer(["ahoj"])
    test_data = Dataset(tok)
    dl = DataLoader(test_data, batch_size=1)
    source_encoder(**tok)
