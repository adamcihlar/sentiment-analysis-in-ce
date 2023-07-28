import pandas as pd
from typing import Type, List
import os

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


def main(predict_scale=True):
    model = AdaptiveSentimentClassifier(
        Preprocessor(),
        Tokenizer(),
        Encoder(),
        ClassificationHead,
        Discriminator(),
        Encoder(path_to_finetuned=paths.OUTPUT_PROD_ENCODER),
        paths.OUTPUT_PROD_CLASSIFIER,
    )

    target_ds = ClassificationDataset(None)
    target_ds.read_user_input()

    target_ds.preprocess(model.preprocessor)
    target_ds.tokenize(model.tokenizer)
    target_ds.create_dataset()
    target_ds.create_dataloader(4, False)

    y_pred = model.bulk_predict(target_ds, predict_scale=predict_scale)

    target_df = target_ds.get_predictions(external_anchor_set=False)
    save_pth = os.path.join(paths.OUTPUT_PREDICTIONS, "sentiment_predictions.csv")
    target_df.to_csv(save_pth)
    logger.info(f"Results saved at {save_pth}.")
    pass


if __name__ == "__main__":
    scale_str = input(
        "Return the sentiment score on scale from 0 to 1 or as classes 0, 1, 2 (negative, neutral, positive)?  (scale/class)\n"
    )
    if scale_str == "scale":
        scale = True
    elif scale_str == "class":
        scale = False
    else:
        raise ValueError('Input must be "scale" or "class".')
    main(predict_scale=scale)
