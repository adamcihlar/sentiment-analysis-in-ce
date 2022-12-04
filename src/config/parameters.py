RANDOM_STATE = 42

# Common model params
FINETUNED_CHECKPOINT = "ufalrobeczech-base_20221127-171131"
FINETUNED_EPOCH = 3
FINETUNED_DATASET = "mall"

# TOKENIZER parameters
class TokenizerParams:
    # TOKENIZER_MODEL = "ufal/robeczech-base"
    MODEL = "Seznam/small-e-czech"
    MAX_LENGTH = 512
    COMBINED_TRUNCATION = True


# DATALOADER parameters
class DataLoaderParams:
    BATCH_SIZE = 64
    SHUFFLE = True
    NUM_WORKERS = 0


# ENCODER parameters
class EncoderParams:
    # MODEL = "ufal/robeczech-base"
    MODEL = "Seznam/small-e-czech"
    FINETUNED_CHECKPOINT = FINETUNED_CHECKPOINT
    FINETUNED_EPOCH = FINETUNED_EPOCH


# CLASSIFIER parameters
class ClassifierParams:
    FINETUNED_CHECKPOINT = FINETUNED_CHECKPOINT
    FINETUNED_EPOCH = FINETUNED_EPOCH
    FINETUNED_DATASET = FINETUNED_DATASET
    INPUT_SIZE = 256
    HIDDEN_SIZE = 256


# OPTIMIZATION parameters
class FinetuningOptimizationParams:
    OPTIMIZATION = {"lr": 2e-5, "betas": (0.9, 0.999)}
    LR_DECAY = 0.95
    WARM_UP_STEPS_PROPORTION = 0.1
    NUM_EPOCHS = 4


class AdaptationOptimizationParams:
    OPTIMIZATION = {"lr": 2e-5, "betas": (0.9, 0.999)}
    LR_DECAY = 0.9
    WARM_UP_STEPS_PROPORTION = 0.1
    NUM_EPOCHS = 4
    TEMPERATURE = 2
    LOSS_COMBINATION_PARAMS = (0.5, 0.5)
