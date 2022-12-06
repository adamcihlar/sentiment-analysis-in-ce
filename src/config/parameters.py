RANDOM_STATE = 42

# Common model params
FINETUNED_CHECKPOINT = "seznamsmall-e-czech_20221204-221123"
FINETUNED_EPOCH = 3
FINETUNED_DATASET = "mall"
ADAPTED_CHECKPOINT = "seznamsmall-e-czech_20221204-221123"
ADAPTED_EPOCH = 3

# DATASET parameters
class DatasetParams:
    BALANCE_DATA = True
    MAJORITY_RATIO = 1


# TOKENIZER parameters
class TokenizerParams:
    # TOKENIZER_MODEL = "ufal/robeczech-base"
    MODEL = "Seznam/small-e-czech"
    MAX_LENGTH = 512
    COMBINED_TRUNCATION = True


# DATALOADER parameters
class DataLoaderParams:
    BATCH_SIZE = 24
    SHUFFLE = True
    NUM_WORKERS = 0


# ENCODER parameters
class EncoderParams:
    # MODEL = "ufal/robeczech-base"
    MODEL = "Seznam/small-e-czech"
    FINETUNED_CHECKPOINT = FINETUNED_CHECKPOINT
    FINETUNED_EPOCH = FINETUNED_EPOCH
    ADAPTED_CHECKPOINT = ADAPTED_CHECKPOINT
    ADAPTED_EPOCH = ADAPTED_EPOCH


# CLASSIFIER parameters
class ClassifierParams:
    FINETUNED_CHECKPOINT = FINETUNED_CHECKPOINT
    FINETUNED_EPOCH = FINETUNED_EPOCH
    FINETUNED_DATASET = FINETUNED_DATASET
    INPUT_SIZE = 256
    HIDDEN_SIZE = 1024


# DISCRIMINATOR parameters
class DiscriminatorParams:
    INPUT_SIZE = 256
    HIDDEN_SIZE = 1024


# OPTIMIZATION parameters
class FinetuningOptimizationParams:
    OPTIMIZATION = {"lr": 2e-5, "betas": (0.9, 0.999)}
    LR_DECAY = 0.95
    WARM_UP_STEPS_PROPORTION = 0.1
    NUM_EPOCHS = 4
    VALIDATION = False
    MIN_QUERY_LEN = 5


class AdaptationOptimizationParams:
    OPTIMIZATION = {"lr": 2e-5, "betas": (0.9, 0.999)}
    LR_DECAY = 0.9
    WARM_UP_STEPS_PROPORTION = 0.1
    NUM_EPOCHS = 4
    TEMPERATURE = 2
    LOSS_COMBINATION_PARAMS = (0.8, 0.2)
    VALIDATION = False


# INFERENCE parameters
class InferenceParams:
    PREDICT_PROBS = True
    TEMPERATURE = 1
