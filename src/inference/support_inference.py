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


def main(predict_scale=True, external_anchor_set=False, anchor_set_size=None):
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

    if not external_anchor_set:
        target_ds.preprocess(model.preprocessor)
        target_ds.tokenize(model.tokenizer)
        target_ds.create_dataset()
        target_ds.create_dataloader(4, False)

        model.suggest_anchor_set(
            target_ds,
            layer=sip.LAYER,
            dim_size=sip.DIM_SIZE,
            anchor_set_size=anchor_set_size,
        )
        input(
            "After labelling the data and saving the file, hit ENTER to continue the evaluation.\n"
        )

    target_ds.read_anchor_set(external_anchor_set)

    if external_anchor_set:
        target_ds.preprocess(model.preprocessor)
        target_ds.tokenize(model.tokenizer)
        target_ds.create_dataset()
        target_ds.create_dataloader(4, False)

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

    target_df = target_ds.get_predictions(external_anchor_set=external_anchor_set)
    save_pth = os.path.join(
        paths.OUTPUT_PREDICTIONS, "supported_sentiment_predictions.csv"
    )
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

    ext_anch_str = input(
        f"Do you have your own anchor set saved at {paths.INPUT_ANCHOR}?  (y/n)\n"
    )
    if ext_anch_str == "y":
        external_anchor_set = True
        ext_anch_size = None
    elif ext_anch_str == "n":
        external_anchor_set = False
        ext_anch_size = input(
            f"Set the size of the anchor set - number of samples to suggest for labelling.\n"
        )
        ext_anch_size = int(ext_anch_size)
    else:
        raise ValueError('Input must be "y" or "n".')

    main(
        predict_scale=scale,
        external_anchor_set=external_anchor_set,
        anchor_set_size=ext_anch_size,
    )
