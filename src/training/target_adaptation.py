from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.config import paths
from src.reading.readers import (
    read_facebook,
    read_mall,
    read_csfd,
    read_finetuning_source,
)
from src.utils.text_preprocessing import Preprocessor
from src.model.tokenizers import Tokenizer
from src.model.encoders import Encoder
from src.model.classifiers import (
    AdaptiveSentimentClassifier,
    ClassificationHead,
    Discriminator,
)

source_train_dataset, source_val_dataset = read_finetuning_source()
target_dataset = read_csfd().sample(10)

asc = AdaptiveSentimentClassifier(
    Preprocessor(),
    Tokenizer(),
    Encoder(path_to_finetuned=paths.OUTPUT_MODELS_FINETUNED_ENCODER_FINAL),
    ClassificationHead,
    Discriminator(),
    Encoder(path_to_finetuned=paths.OUTPUT_MODELS_FINETUNED_ENCODER_FINAL),
    # paths.OUTPUT_MODELS_FINETUNED_CLASSIFIER_FINAL,
)

type(acs.classifier)
