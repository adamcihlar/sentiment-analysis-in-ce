import os
import pandas as pd
from src.config import paths
from src.config.parameters import SupportInferenceParams as sip
from src.utils.datasets import ClassificationDataset, drop_undefined_classes
from src.model.classifiers import (
    AdaptiveSentimentClassifier,
    ClassificationHead,
    Discriminator,
)
from src.model.encoders import Encoder
from src.model.tokenizers import Tokenizer
from src.utils.text_preprocessing import Preprocessor
from loguru import logger


def main(scale):
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

    logger.info("Getting the semantic representations for your input data")
    model.bulk_predict(target_ds, predict_scale=True, output_hidden=sip.LAYER)

    model.suggest_anchor_set(
        target_ds,
        layer=sip.LAYER,
        dim_size=sip.DIM_SIZE,
        anchor_set_size=sip.SUPPORT_SET_SIZE,
    )
    input(
        "After labelling the data and saving the file, hit ENTER to continue the evaluation.\n"
    )

    target_ds.read_anchor_set()

    y_pred = model.mix_bulk_predict(
        target_ds,
        knn=1,
        layer=sip.LAYER,
        dim_size=sip.DIM_SIZE,
        k=sip.K,
        scale=scale,
        emp_prob=sip.EMP_CONF,
        radius=True,
    )

    target_df = target_ds.get_predictions()
    save_pth = os.path.join(paths.OUTPUT_PREDICTIONS, "emails.csv")
    test_df.to_csv(save_pth)
    logger.info(f"Results saved at {save_pth}.")
    pass


if __name__ == "__main__":
    main(True)