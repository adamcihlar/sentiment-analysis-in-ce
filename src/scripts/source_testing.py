import os
import pandas as pd
from src.config import paths
from src.utils.datasets import ClassificationDataset, drop_undefined_classes
from src.model.classifiers import (
    AdaptiveSentimentClassifier,
    ClassificationHead,
    Discriminator,
)
from src.model.encoders import Encoder
from src.model.tokenizers import Tokenizer
from src.utils.text_preprocessing import Preprocessor

if __name__ == "__main__":

    test_sets = ["facebook", "mall", "csfd"]

    models = [
        [
            ("seznamsmall-e-czech_20230123-173904", "5", "facebook", "ordinal"),
            ("seznamsmall-e-czech_20230124-091654", "3", "facebook", "multiclass"),
            ("seznamsmall-e-czech_20230126-195853", "2", "facebook", "ordinal"),
            ("seznamsmall-e-czech_20230127-220848", "5", "facebook", "multiclass"),
            ("seznamsmall-e-czech_20230128-144144", "4", "facebook", "ordinal"),
            ("seznamsmall-e-czech_20230128-171700", "5", "facebook", "ordinal"),
            (
                "seznamsmall-e-czech_20230218-071534",
                "5",
                "csfd_facebook_mall",
                "ordinal",
            ),
            ("seznamsmall-e-czech_20230218-142950", "5", "mall_facebook", "ordinal"),
            ("seznamsmall-e-czech_20230218-183829", "5", "facebook_csfd", "ordinal"),
        ],
        [
            ("seznamsmall-e-czech_20230123-213412", "5", "mall", "ordinal"),
            ("seznamsmall-e-czech_20230124-120240", "5", "mall", "multiclass"),
            ("seznamsmall-e-czech_20230126-195853", "5", "mall", "ordinal"),
            ("seznamsmall-e-czech_20230127-220848", "5", "mall", "multiclass"),
            ("seznamsmall-e-czech_20230128-055746", "5", "mall", "ordinal"),
            ("seznamsmall-e-czech_20230128-171700", "5", "mall", "ordinal"),
            (
                "seznamsmall-e-czech_20230218-071534",
                "5",
                "csfd_facebook_mall",
                "ordinal",
            ),
            ("seznamsmall-e-czech_20230218-142950", "5", "mall_facebook", "ordinal"),
            ("seznamsmall-e-czech_20230218-213739", "5", "mall_csfd", "ordinal"),
        ],
        [
            ("seznamsmall-e-czech_20230123-181226", "5", "csfd", "ordinal"),
            ("seznamsmall-e-czech_20230124-094038", "5", "csfd", "multiclass"),
            ("seznamsmall-e-czech_20230126-195853", "4", "csfd", "ordinal"),
            ("seznamsmall-e-czech_20230127-220848", "5", "csfd", "multiclass"),
            ("seznamsmall-e-czech_20230128-055746", "5", "csfd", "ordinal"),
            ("seznamsmall-e-czech_20230128-144144", "5", "csfd", "ordinal"),
            (
                "seznamsmall-e-czech_20230218-071534",
                "5",
                "csfd_facebook_mall",
                "ordinal",
            ),
            ("seznamsmall-e-czech_20230218-183829", "5", "facebook_csfd", "ordinal"),
            ("seznamsmall-e-czech_20230218-213739", "5", "mall_csfd", "ordinal"),
        ],
    ]

    for i, test_set in enumerate(test_sets):
        test_path = os.path.join(paths.DATA_FINAL_FINETUNING_TEST, test_set + ".csv")
        test_df = pd.read_csv(test_path, index_col=0)

        test_df = drop_undefined_classes(test_df)

        for model in models[i]:

            enc = "_".join([model[0], model[1]])
            enc_pth = os.path.join(paths.OUTPUT_MODELS_FINETUNED_ENCODER, enc)
            cls = "_".join([model[0], model[2], model[1]])
            cls_pth = os.path.join(paths.OUTPUT_MODELS_FINETUNED_CLASSIFIER, cls)
            file_name = (
                test_set + "_" + "_".join([model[0], model[2], model[1]]) + ".json"
            )
            save_path = os.path.join(paths.OUTPUT_INFO_FINETUNING, "test", file_name)

            asc = AdaptiveSentimentClassifier(
                Preprocessor(),
                Tokenizer(),
                Encoder(path_to_finetuned=enc_pth),
                ClassificationHead,
                Discriminator(),
                Encoder(path_to_finetuned=enc_pth),
                classifier_checkpoint_path=cls_pth,
                inference_mode=True,
                task_settings=model[3],
            )

            test = ClassificationDataset(test_df.text, test_df.label, test_df.source)
            test.preprocess(asc.preprocessor)
            test.tokenize(asc.tokenizer)
            test.create_dataset()
            test.create_dataloader(16, False)

            y_pred = asc.bulk_predict(test, predict_scale=False)
            test.evaluation_report(save_path)
