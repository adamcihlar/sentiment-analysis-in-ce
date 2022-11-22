import os

# DATA
DATA = "data"

# RAW
DATA_RAW = os.path.join(DATA, "raw")

DATA_RAW_ZIP_FACEBOOK = os.path.join(DATA_RAW, "facebook.zip")
DATA_RAW_ZIP_CSFD = os.path.join(DATA_RAW, "csfd.zip")
DATA_RAW_ZIP_MALL = os.path.join(DATA_RAW, "mall.zip")

DATA_RAW_DIR_FACEBOOK = os.path.join(DATA_RAW, "facebook")
DATA_RAW_DIR_CSFD = os.path.join(DATA_RAW, "csfd")
DATA_RAW_DIR_MALL = os.path.join(DATA_RAW, "mall")

# PROCESSED
DATA_PROCESSED = os.path.join(DATA, "preprocessed")

DATA_PROCESSED_FACEBOOK = os.path.join(DATA_PROCESSED, "facebook.csv")
DATA_PROCESSED_CSFD = os.path.join(DATA_PROCESSED, "csfd.csv")
DATA_PROCESSED_MALL = os.path.join(DATA_PROCESSED, "mall.csv")

DATA_PROCESSED_CONCAT = os.path.join(DATA_PROCESSED, "concat_dataset.csv")

# OUTPUT
OUTPUT = "output"

# PREDICTIONS
OUTPUT_PREDICTIONS = os.path.join(OUTPUT, "predictions")

# MODELS
OUTPUT_MODELS = os.path.join(OUTPUT, "models")

# FINETUNNED
OUTPUT_MODELS_FINETUNNED = os.path.join(OUTPUT_MODELS, "finetunned")
OUTPUT_MODELS_FINETUNNED_ENCODER = os.path.join(OUTPUT_MODELS_FINETUNNED, "encoder")
OUTPUT_MODELS_FINETUNNED_CLASSIFIER = os.path.join(
    OUTPUT_MODELS_FINETUNNED, "classification_head"
)

# ADAPTED
OUTPUT_MODELS_ADAPTED = os.path.join(OUTPUT_MODELS, "adapted")
OUTPUT_MODELS_ADAPTED_ENCODER = os.path.join(OUTPUT_MODELS_ADAPTED, "encoder")
OUTPUT_MODELS_ADAPTED_DISCRIMINATOR = os.path.join(
    OUTPUT_MODELS_ADAPTED, "discriminator"
)

# TRAIN INFO
OUTPUT_INFO = os.path.join(OUTPUT, "train_info")

# FINETUNNING
OUTPUT_INFO_FINETUNNING = os.path.join(OUTPUT_INFO, "finetunning")
# ADAPTATION
OUTPUT_INFO_ADAPTATION = os.path.join(OUTPUT_INFO, "adaptation")
