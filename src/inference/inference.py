import pandas as pd
from typing import Type, List

from src.config import paths
from src.utils.datasets import ClassificationDataset
from src.model.classifiers import AdaptiveSentimentClassifier
from src.utils.text_preprocessing import Preprocessor
from src.model.tokenizers import Tokenizer
from src.model.encoders import Encoder
from src.model.classifiers import (
    AdaptiveSentimentClassifier,
    ClassificationHead,
    Discriminator,
)


def inference(
    model: Type[AdaptiveSentimentClassifier],
    texts: List[str],
    predict_probs: bool = True,
    temperature: int = 1,
    labels: List[int] = None,
    return_input: bool = False,
) -> pd.DataFrame:
    if len(texts) <= 16:
        pred = model.predict(texts, predict_probs, temperature)
    else:
        dataset = ClassificationDataset(texts, labels)
        dataset.preprocess(model.preprocessor)
        dataset.tokenize(model.tokenizer)
        dataset.create_dataset()
        dataset.create_dataloader(16, False, 0)
        model.bulk_predict(dataset)
    return dataset.get_predictions(return_input)


if __name__ == "__main__":
    texts = [
        "tohle smrdi",
        "tvoje mam je zidlicka",
        "miluju zidlicky",
        "uplne k nicemu",
        "tohle smrdi",
        "tohle smrdi",
        "tvoje mam je zidlicka",
        "miluju zidlicky",
        "uplne k nicemu",
        "tvoje mam je zidlicka",
        "miluju zidlicky",
        "uplne k nicemu",
        "tohle smrdi",
        "tohle smrdi",
        "tohle smrdi",
        "tohle smrdi",
    ]

    dataset = ClassificationDataset(texts)

    asc = AdaptiveSentimentClassifier(
        Preprocessor(),
        Tokenizer(),
        Encoder(),
        ClassificationHead,
        Discriminator(),
        Encoder(path_to_finetuned=paths.OUTPUT_MODELS_ADAPTED_ENCODER_FINAL),
        paths.OUTPUT_MODELS_FINETUNED_CLASSIFIER_FINAL,
    )

    model = asc

    asc.predict(texts)
