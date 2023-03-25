import os

from src.config import parameters

# DATA
DATA = "data"

# RAW
DATA_RAW = os.path.join(DATA, "raw")

DATA_RAW_ZIP_FACEBOOK = os.path.join(DATA_RAW, "facebook.zip")
DATA_RAW_ZIP_CSFD = os.path.join(DATA_RAW, "csfd.zip")
DATA_RAW_ZIP_MALL = os.path.join(DATA_RAW, "mall.zip")
DATA_RAW_ZIP_EMAILS = os.path.join(DATA_RAW, "mails.zip")

DATA_RAW_DIR_FACEBOOK = os.path.join(DATA_RAW, "facebook")
DATA_RAW_DIR_CSFD = os.path.join(DATA_RAW, "csfd")
DATA_RAW_DIR_MALL = os.path.join(DATA_RAW, "mall")
DATA_RAW_DIR_EMAILS = os.path.join(DATA_RAW, "mails")
DATA_RAW_RESPONSES = os.path.join(DATA_RAW, "responses_2023-02-03.RData")
DATA_RAW_SENT = os.path.join(DATA_RAW, "sent_2023-02-03.RData")

# PROCESSED
DATA_PROCESSED = os.path.join(DATA, "preprocessed")

DATA_PROCESSED_FACEBOOK = os.path.join(DATA_PROCESSED, "facebook.csv")
DATA_PROCESSED_CSFD = os.path.join(DATA_PROCESSED, "csfd.csv")
DATA_PROCESSED_MALL = os.path.join(DATA_PROCESSED, "mall.csv")
DATA_PROCESSED_RESPONSES = os.path.join(DATA_PROCESSED, "responses.csv")
DATA_PROCESSED_RESPONSES_CONFIRMED = os.path.join(
    DATA_PROCESSED, "responses_confirmed.csv"
)

DATA_PROCESSED_CONCAT = os.path.join(DATA_PROCESSED, "concat_dataset.csv")

# FINAL
DATA_FINAL = os.path.join(DATA, "final")

DATA_FINAL_SOURCE_TRAIN = os.path.join(DATA_FINAL, "source_train")
DATA_FINAL_SOURCE_VAL = os.path.join(DATA_FINAL, "source_val")

DATA_FINAL_FINETUNING_TRAIN = os.path.join(DATA_FINAL, "finetuning_train")
DATA_FINAL_FINETUNING_VAL = os.path.join(DATA_FINAL, "finetuning_val")
DATA_FINAL_FINETUNING_TEST = os.path.join(DATA_FINAL, "finetuning_test")

DATA_FINAL_ADAPTATION_TARGET = os.path.join(DATA_FINAL, "adaptation_target")

# OUTPUT
OUTPUT = "output"

# PREDICTIONS
OUTPUT_PREDICTIONS = os.path.join(OUTPUT, "predictions")

# MODELS
OUTPUT_MODELS = os.path.join(OUTPUT, "models")

# FINETUNED
OUTPUT_MODELS_FINETUNED = os.path.join(OUTPUT_MODELS, "finetuned")
OUTPUT_MODELS_FINETUNED_ENCODER = os.path.join(OUTPUT_MODELS_FINETUNED, "encoder")
OUTPUT_MODELS_FINETUNED_CLASSIFIER = os.path.join(
    OUTPUT_MODELS_FINETUNED, "classification_head"
)
OUTPUT_MODELS_FINETUNED_ENCODER_FINAL = os.path.join(
    OUTPUT_MODELS_FINETUNED_ENCODER,
    "_".join([parameters.FINETUNED_CHECKPOINT, str(parameters.FINETUNED_EPOCH)]),
)
OUTPUT_MODELS_FINETUNED_CLASSIFIER_FINAL = os.path.join(
    OUTPUT_MODELS_FINETUNED_CLASSIFIER,
    "_".join(
        [
            parameters.FINETUNED_CHECKPOINT,
            parameters.FINETUNED_DATASET,
            str(parameters.FINETUNED_EPOCH),
        ]
    ),
)

# ADAPTED
OUTPUT_MODELS_ADAPTED = os.path.join(OUTPUT_MODELS, "adapted")
OUTPUT_MODELS_ADAPTED_ENCODER = os.path.join(OUTPUT_MODELS_ADAPTED, "encoder")
OUTPUT_MODELS_ADAPTED_DISCRIMINATOR = os.path.join(
    OUTPUT_MODELS_ADAPTED, "discriminator"
)
OUTPUT_MODELS_ADAPTED_ENCODER_FINAL = os.path.join(
    OUTPUT_MODELS_ADAPTED_ENCODER,
    "_".join([parameters.ADAPTED_CHECKPOINT, str(parameters.ADAPTED_EPOCH)]),
)

# TRAIN INFO
OUTPUT_INFO = os.path.join(OUTPUT, "train_info")

# FINETUNING
OUTPUT_INFO_FINETUNING = os.path.join(OUTPUT_INFO, "finetuning")
# ADAPTATION
OUTPUT_INFO_ADAPTATION = os.path.join(OUTPUT_INFO, "adaptation")
OUTPUT_INFO_ADAPTATION_MALL = os.path.join(OUTPUT_INFO_ADAPTATION, "mall")
OUTPUT_INFO_ADAPTATION_CSFD = os.path.join(OUTPUT_INFO_ADAPTATION, "csfd")
OUTPUT_INFO_ADAPTATION_FACEBOOK = os.path.join(OUTPUT_INFO_ADAPTATION, "facebook")

# APP
OUTPUT_PROD = os.path.join(OUTPUT, "prod")
OUTPUT_PROD_ENCODER = os.path.join(OUTPUT_PROD, "encoder")
OUTPUT_PROD_CLASSIFIER = os.path.join(OUTPUT_PROD, "classifier")

# ASSETS
OUTPUT_ASSETS = os.path.join(OUTPUT, "assets")
OUTPUT_ASSETS_TEST_REPORTS = os.path.join(OUTPUT_ASSETS, "test_reports")
