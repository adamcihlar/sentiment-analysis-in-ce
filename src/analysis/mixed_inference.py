import os
import re
import numpy as np
import pandas as pd
from src.config import paths
import itertools

import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder


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
            name = filename.split(".json")[0]
            name = name.split(dataset + "_")[1]
            split_name = name.split("_")
            df["ds"] = dataset
            df["layer"] = split_name[0]
            df["knn"] = split_name[1]
            df["pca"] = split_name[2]
            df["anchor_size"] = split_name[3]
            if len(split_name) > 4:
                df["empirical_conf"] = split_name[4]
            else:
                df["empirical_conf"] = False

            test_results_df = pd.concat([test_results_df, df], axis=0)

        test_results_df = test_results_df.loc["macro avg"]

        save_path = os.path.join(path_to_results, "summary", dataset + ".csv")
        test_results_df.to_csv(save_path)
    return test_results_df


def regression_to_analyze_mixed_inference_with_1nn():
    mall = pd.read_csv("output/train_info/inference/summary/mall.csv", index_col=0)
    base_mall = pd.read_csv(
        "output/train_info/inference/summary/base_mall.csv", index_col=0
    )
    kmeans_mall = pd.read_csv(
        "output/train_info/inference/summary/kmeans_mall.csv", index_col=0
    )
    kmeans_mix_mall = pd.read_csv(
        "output/train_info/inference/summary/kmeans_mix_mall.csv", index_col=0
    )

    facebook = pd.read_csv(
        "output/train_info/inference/summary/facebook.csv", index_col=0
    )
    base_facebook = pd.read_csv(
        "output/train_info/inference/summary/base_facebook.csv", index_col=0
    )
    kmeans_facebook = pd.read_csv(
        "output/train_info/inference/summary/kmeans_facebook.csv", index_col=0
    )
    kmeans_mix_facebook = pd.read_csv(
        "output/train_info/inference/summary/kmeans_mix_facebook.csv", index_col=0
    )

    csfd = pd.read_csv("output/train_info/inference/summary/csfd.csv", index_col=0)
    base_csfd = pd.read_csv(
        "output/train_info/inference/summary/base_csfd.csv", index_col=0
    )
    kmeans_csfd = pd.read_csv(
        "output/train_info/inference/summary/kmeans_csfd.csv", index_col=0
    )
    kmeans_mix_csfd = pd.read_csv(
        "output/train_info/inference/summary/kmeans_mix_csfd.csv", index_col=0
    )

    hdbscan = pd.concat([mall, csfd, facebook])
    hdbscan["clustering"] = "hdbscan"

    kmeans = pd.concat([kmeans_mix_mall, kmeans_mix_csfd, kmeans_mix_facebook])
    kmeans["clustering"] = "kmeans"

    ds = pd.concat([hdbscan, kmeans])
    ds["mix"] = 1

    base = pd.concat([base_mall, base_csfd, base_facebook])
    base["clustering"] = "hdbscan"

    base_kmeans = pd.concat([kmeans_mall, kmeans_csfd, kmeans_facebook])
    base_kmeans["clustering"] = "kmeans"
    dsb = pd.concat([base, base_kmeans])
    dsb["mix"] = 0

    ds = pd.concat([ds, dsb])

    ds = ds.drop(columns=["mae", "rmse"])
    y = ds.iloc[:, 0]
    X = ds.iloc[:, 1:]

    ohc_cols = []
    ohc_col_names = []
    for col in X.columns:
        uniq = X[col].unique()[1:]
        for val in uniq:
            ohc_cols.append((X[col] == val) * 1)
            ohc_col_names.append(val)
    X_tr = pd.concat(ohc_cols, axis=1)
    X_tr.columns = ohc_col_names
    X_tr = sm.add_constant(X_tr)

    mod = sm.OLS(y, X_tr).fit()
    print(mod.summary())
    pass


if __name__ == "__main__":
    hdbscan_datasets = ["facebook", "mall", "csfd"]
    hdbscan_mix = main(hdbscan_datasets)

    hdbscan_base_datasets = ["base_facebook", "base_mall", "base_csfd"]
    hdbscan_base = main(hdbscan_base_datasets)

    kmeans_datasets = ["kmeans_mix_facebook", "kmeans_mix_mall", "kmeans_mix_csfd"]
    kmeans_mix = main(kmeans_datasets)

    kmeans_base_datasets = ["kmeans_facebook", "kmeans_mall", "kmeans_csfd"]
    kmeans_base = main(kmeans_base_datasets)

    knn_datasets = ["knn_mix_facebook", "knn_mix_mall", "knn_mix_csfd"]
    knn_mix = main(knn_datasets)

    # linear regression
    knn_mix_mall = pd.read_csv(
        "output/train_info/inference/summary/knn_mix_mall.csv", index_col=0
    )
    knn_mix_facebook = pd.read_csv(
        "output/train_info/inference/summary/knn_mix_facebook.csv", index_col=0
    )
    knn_mix_csfd = pd.read_csv(
        "output/train_info/inference/summary/knn_mix_csfd.csv", index_col=0
    )

    ds = pd.concat([knn_mix_csfd, knn_mix_mall, knn_mix_facebook])

    ds = ds.drop(columns=["mae", "rmse"])
    # ds = ds.drop(columns=["mae", "rmse", 'ds'])

    ds = ds.loc[ds.anchor_size==100]
    ds = ds.loc[ds.ds=='knn_mix_facebook']
    ds = ds.drop(columns='ds')

    y = ds.iloc[:, 0]
    X = ds.iloc[:, 1:]

    ohc_cols = []
    ohc_col_names = []
    for col in X.columns:
        uniq = X[col].unique()[1:]
        for val in uniq:
            ohc_cols.append((X[col] == val) * 1)
            ohc_col_names.append(val)
    X_tr = pd.concat(ohc_cols, axis=1)
    X_tr.columns = ohc_col_names
    X_tr = sm.add_constant(X_tr)

    mod = sm.OLS(y, X_tr).fit()
    print(mod.summary())
