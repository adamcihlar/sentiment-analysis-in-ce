dowload_raw_data:
	python -m src.loading.download_data

get_raw_data: 
	python -m src.loading.get_raw_data

preprocess_raw_data: get_raw_data
	python -m src.preprocessing.preprocess_csfd
	python -m src.preprocessing.preprocess_mall
	python -m src.preprocessing.preprocess_facebook

get_concatenated_dataset: preprocess_raw_data
	python -m src.preprocessing.concatenate_datasets
