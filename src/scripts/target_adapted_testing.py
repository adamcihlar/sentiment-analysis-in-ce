import os
import pandas as pd
from src.config import paths, parameters
from src.utils.datasets import ClassificationDataset, drop_undefined_classes
from src.model.classifiers import (
    AdaptiveSentimentClassifier,
    ClassificationHead,
    Discriminator,
)
from src.model.encoders import Encoder
from src.model.tokenizers import Tokenizer
from src.utils.text_preprocessing import Preprocessor
from src.reading.readers import (
    read_csfd,
    read_facebook,
    read_mall,
)

if __name__ == "__main__":
    test_sets = ["mall", "csfd", "facebook"]

    # mall
    mall_models = [
        ("seznamsmall-e-czech_20230123-173904", "5", "facebook"),
        ("seznamsmall-e-czech_20230123-181226", "5", "csfd"),
        ("seznamsmall-e-czech_20230128-144144", "4", "facebook"),
        ("seznamsmall-e-czech_20230128-144144", "5", "csfd"),
        ("seznamsmall-e-czech_20230218-183829", "5", "facebook_csfd"),
    ]

    # csfd
    csfd_models = [
        ("seznamsmall-e-czech_20230123-173904", "5", "facebook"),
        ("seznamsmall-e-czech_20230123-213412", "5", "mall"),
        ("seznamsmall-e-czech_20230128-171700", "5", "facebook"),
        ("seznamsmall-e-czech_20230128-171700", "5", "mall"),
        ("seznamsmall-e-czech_20230218-142950", "5", "mall_facebook"),
    ]

    # facebook
    facebook_models = [
        ("seznamsmall-e-czech_20230123-181226", "5", "csfd"),
        ("seznamsmall-e-czech_20230123-213412", "5", "mall"),
        ("seznamsmall-e-czech_20230128-055746", "5", "csfd"),
        ("seznamsmall-e-czech_20230128-055746", "5", "mall"),
        ("seznamsmall-e-czech_20230218-213739", "5", "mall_csfd"),
    ]

    models = [mall_models] + [csfd_models] + [facebook_models]

    datasets = [read_mall] + [read_csfd] + [read_facebook]

    for i, test_set in enumerate(test_sets):

        test_df = datasets[i]()
        test_df = drop_undefined_classes(test_df)
        test_df = test_df.sample(parameters.AdaptationOptimizationParams.N_EMAILS)

        for model in models[i]:

            enc = "_".join([model[0], model[1]])
            enc_pth = os.path.join(paths.OUTPUT_MODELS_FINETUNED_ENCODER, enc)
            cls = "_".join([model[0], model[2], model[1]])
            cls_pth = os.path.join(paths.OUTPUT_MODELS_FINETUNED_CLASSIFIER, cls)
            file_name = (
                test_set + "_" + "_".join([model[0], model[2], model[1]]) + ".json"
            )
            save_path = os.path.join(paths.OUTPUT_INFO_ZEROSHOT, file_name)

            asc = AdaptiveSentimentClassifier(
                Preprocessor(),
                Tokenizer(),
                Encoder(path_to_finetuned=enc_pth),
                ClassificationHead,
                Discriminator(),
                Encoder(path_to_finetuned=enc_pth),
                classifier_checkpoint_path=cls_pth,
                inference_mode=True,
                task_settings="ordinal",
            )

            test = ClassificationDataset(test_df.text, test_df.label, test_df.source)
            test.preprocess(asc.preprocessor)
            test.tokenize(asc.tokenizer)
            test.create_dataset()
            test.create_dataloader(16, False)

            y_pred = asc.bulk_predict(test, predict_scale=False)
            test.evaluation_report(save_path)
