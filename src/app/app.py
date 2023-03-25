import fire
from flask import Flask, render_template, request

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

app = Flask(__name__)


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

    @app.route("/", methods=["POST", "GET"])
    def form():
        if request.method == "GET":
            return render_template("input.html")
        if request.method == "POST":
            form_data = request.form
            query = form_data["Query"]
            label = round(model.predict([query])[0], 3)
            return render_template("input.html", label=label, query=query)

    app.run(host="0.0.0.0", port=8081)


if __name__ == "__main__":
    fire.Fire(main)
