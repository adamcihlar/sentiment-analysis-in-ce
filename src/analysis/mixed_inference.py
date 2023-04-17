import os
import re
import numpy as np
import pandas as pd
from src.config import paths
import itertools


def main(datasets):
    """
    Expects that models were already finetuned and there are results on test
    sets available.
    Loads all the classification reports for given test set (one per model) and
    concatenates them together to one csv and adds some metadata about the
    model, saves the csv.
    (Expects that models were trained in particular order - the metadata are
    hardcoded).
    """
    for dataset in datasets:
        path_to_results = paths.OUTPUT_INFO_INFERENCE
        re_pattern = "^" + dataset

        files = os.listdir(path_to_results)
        jsons_mask = np.array([bool(re.search(".json$", file)) for file in files])
        dataset_mask = np.array([bool(re.search(re_pattern, file)) for file in files])
        files_mask = jsons_mask & dataset_mask
        test_files = np.array(files)[files_mask]
        test_files = sorted(test_files)

        test_results_df = pd.DataFrame()
        for i, filename in enumerate(test_files):
            f = os.path.join(path_to_results, filename)

            df = (
                pd.read_json(f)
                .loc[["f1-score", "mae", "rmse"]]
                .dropna(axis=1)
                .drop(columns="accuracy")
            ).T
            name = filename.split(".")[0]
            split_name = name.split("_")
            df["layer"] = split_name[1]
            df["knn"] = split_name[2]
            df["pca"] = split_name[3]
            df["anchor_size"] = split_name[4]

            test_results_df = pd.concat([test_results_df, df], axis=0)

        test_results_df = test_results_df.loc["macro avg"]

        save_path = os.path.join(path_to_results, dataset + ".csv")
        test_results_df.to_csv(save_path)
    return test_results_df


if __name__ == "__main__":
    datasets = ["facebook", "mall", "csfd"]
    mix = main(datasets)
    base_datasets = ["base_facebook", "base_mall", "base_csfd"]
    base = main(base_datasets)

    mall = pd.read_csv("output/train_info/inference/mall.csv", index_col=0)
    facebook = pd.read_csv("output/train_info/inference/facebook.csv", index_col=0)
    csfd = pd.read_csv("output/train_info/inference/csfd.csv", index_col=0)
