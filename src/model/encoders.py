from loguru import logger
from transformers import RobertaModel
from src.config.parameters import ENCODER_MODEL
from torch import nn


class Encoder(nn.Module):
    """
    Easy initialization of the correct encoder.
    Simplification of the forward method to output only the embeddings.
    """

    def __init__(self, path_to_finetuned=None):
        super(Encoder, self).__init__()
        self.encoder = RobertaModel.from_pretrained(ENCODER_MODEL)
        if path_to_finetuned is not None:
            logger.info(f"Loading model parameters from {path_to_finetuned}.")
            self.encoder.load_state_dict(torch.load(path_to_finetuned))

    def forward(self, x, mask=None):
        outputs = self.encoder(x, attention_mask=mask)
        sequence_output = outputs[0]
        features = sequence_output[:, 0, :]
        return features
