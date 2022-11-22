from src.reading.readers import read_facebook, read_mall, read_csfd

source_train_dataset = read_mall().sample(10)
source_val_dataset = read_mall().sample(4)
target_dataset = read_csfd().sample(10)

asc = AdaptiveSentimentClassifier(
    Preprocessor(),
    Tokenizer(),
    Encoder(path_to_finetuned=paths.OUTPUT_MODELS_FINETUNED_ENCODER_FINAL),
    ClassificationHead,
    Discriminator(),
    Encoder(path_to_finetuned=paths.OUTPUT_MODELS_FINETUNED_ENCODER_FINAL),
)
