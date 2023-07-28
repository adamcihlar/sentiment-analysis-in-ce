import fire
from fastapi import FastAPI, Response, status
import uvicorn

from src.config import paths
from src.config.parameters import InferenceParams
from src.api.base import SearchRequest
from src.utils.text_preprocessing import Preprocessor
from src.model.tokenizers import Tokenizer
from src.model.encoders import Encoder
from src.model.classifiers import (
    AdaptiveSentimentClassifier,
    ClassificationHead,
    Discriminator,
)

fastapi_app = FastAPI()


def main():
    model = AdaptiveSentimentClassifier(
        Preprocessor(),
        Tokenizer(),
        Encoder(),
        ClassificationHead,
        Discriminator(),
        Encoder(path_to_finetuned=paths.OUTPUT_PROD_ENCODER),
        paths.OUTPUT_PROD_CLASSIFIER,
    )

    @fastapi_app.post("/query")
    def query(request: SearchRequest):
        request = request.dict()
        label = model.predict(
            [request["query"]],
            InferenceParams.PREDICT_PROBS,
            InferenceParams.TEMPERATURE,
        )
        return {"label": label}

    uvicorn.run(app=fastapi_app, host="0.0.0.0", port=8080, workers=1)


if __name__ == "__main__":
    fire.Fire(main)
