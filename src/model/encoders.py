from loguru import logger
from django.utils.text import slugify
from transformers import RobertaModel
from torch import nn

from src.config.parameters import ENCODER_MODEL


class Encoder(nn.Module):
    """
    Easy initialization of the correct encoder.
    Simplification of the forward method to output only the embeddings.
    """

    def __init__(self, model_name=ENCODER_MODEL, path_to_finetuned=None):
        super(Encoder, self).__init__()
        self.encoder = RobertaModel.from_pretrained(model_name)
        self.name = slugify(model_name)
        if path_to_finetuned is not None:
            logger.info(f"Loading model parameters from {path_to_finetuned}.")
            self.encoder.load_state_dict(torch.load(path_to_finetuned))

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        features = sequence_output[:, 0, :]
        return features
