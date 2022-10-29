download_raw_data:
	python -m src.loading.download_data

get_raw_data: 
	python -m src.loading.get_raw_data

preprocess_raw_data: get_raw_data
	python -m src.preprocessing.preprocess_csfd
	python -m src.preprocessing.preprocess_mall
	python -m src.preprocessing.preprocess_facebook

get_concatenated_dataset: preprocess_raw_data
	python -m src.preprocessing.concatenate_datasets

install_requirements:
	pip install -r requirements.txt

install: install_requirements
	pip install -e .

build_image:
	docker image prune -f
	docker build -t sentiment_analysis_in_ce .
	docker image prune -f

run_image:
	docker run --rm -it -v $$PWD:/app -w /app -p 5001:5001 sentiment_analysis_in_ce /bin/bash
