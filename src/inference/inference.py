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
) -> Type[ClassificationDataset]:
    """
    Gets texts (+labels), predicts the scores and returns the whole created ClassificationDataset.
    """
    if len(texts) <= 16:
        pred = model.predict(texts, predict_probs, temperature)
    else:
        dataset = ClassificationDataset(texts, labels)
        dataset.preprocess(model.preprocessor)
        dataset.tokenize(model.tokenizer)
        dataset.create_dataset()
        dataset.create_dataloader(16, False, 0)
        model.bulk_predict(dataset)
    return dataset


def evaluate(model: Type[AdaptiveSentimentClassifier], path_to_test: str):
    """
    Gets path to dataset with labels (test), reads the dataset,
    calls inference and outputs dataset with labels, predictions and metrics and saves evaluation report.
    Returns dataset with labels and predictions.
    """
    test_df = pd.read_csv(path_to_test)
    test_ds = inference(
        model,
        test_df["text"],
        predict_probs=False,
        temperature=1,
        labels=test_df["label"],
    )
    save_path = os.path.join(paths.OUTPUT_ASSETS_TEST_REPORTS, model.full_name)
    os.makedirs(os.path.split(save_path)[0], exist_ok=True)
    test_ds.evaluation_report(save_path=save_path)
    return test_ds


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
        "tvoje mam je zidlicka",
        "miluju zidlicky",
        "uplne k nicemu",
        "tohle smrdi",
        "miluju zidlicky",
        "uplne k nicemu",
        "tohle smrdi",
        "tohle smrdi",
        "tohle jsou nejlepsi zidlicky",
        "tohle smrdi",
        "super!!!",
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
        inference_mode=True,
    )

    inference(asc, texts)
