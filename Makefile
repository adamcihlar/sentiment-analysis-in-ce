# installation
docker_build_inference:
	docker image prune -f
	yes | cp -rf ./docker/inference/Dockerfile Dockerfile
	docker build -t sentiment_analysis_in_ce_inference .
	docker image prune -f
	rm -f Dockerfile

docker_build_api:
	docker image prune -f
	yes | cp -rf ./docker/api/Dockerfile Dockerfile
	docker build -t sentiment_analysis_in_ce_api .
	docker image prune -f
	rm -f Dockerfile

docker_build_app:
	docker image prune -f
	yes | cp -rf ./docker/app/Dockerfile Dockerfile
	docker build -t sentiment_analysis_in_ce_app .
	docker image prune -f
	rm -f Dockerfile

install_requirements:
	pip install -r requirements.txt

install: install_requirements
	pip install -e .

# data science pipeline
download_raw_data:
	python -m src.ingestion.download_data

get_raw_data: 
	python -m src.ingestion.get_raw_data

preprocess_raw_data: get_raw_data
	python -m src.preprocessing.preprocess_csfd
	python -m src.preprocessing.preprocess_mall
	python -m src.preprocessing.preprocess_facebook

get_concatenated_dataset: preprocess_raw_data
	python -m src.preprocessing.concatenate_datasets

preprocess_raw_responses:
	python -m src.preprocessing.preprocess_responses

# development helpers
docker_build_dev:
	docker image prune -f
	yes | cp -rf ./docker/dev/Dockerfile Dockerfile
	docker build -t sentiment_analysis_in_ce_dev .
	docker image prune -f
	rm -f Dockerfile

docker_run_dev:
	docker run --rm -it -v $$PWD:/app -w /app -p 5001:5001 sentiment_analysis_in_ce_dev /bin/bash

docker_connect:
	docker exec -it "$$(docker ps -q)" /bin/bash

clean_finetuning:
	sudo rm -r -i output/models/finetuned
	sudo rm -r -i output/train_info/finetuning

clean_adaptation:
	sudo rm -r -i output/models/adapted
	sudo rm -r -i output/train_info/adapted

# run solution
app:
	python -m src.app.app

run_app: docker_build_app
	docker run --rm -it -v $$PWD:/app -w /app -p 8081:8081 sentiment_analysis_in_ce_app

api:
	python -m src.api.model_api

run_api: docker_build_api
	docker run --rm -it -v $$PWD:/app -w /app -p 5001:5001 sentiment_analysis_in_ce_api /bin/bash

run_inference: docker_build_inference
	docker run --rm -v $$PWD:/app -w /app -p 5001:5001 sentiment_analysis_in_ce_inference /bin/bash
