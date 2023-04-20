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
        path_to_results = paths.OUTPUT_INFO_FINETUNING

        re_pattern = "_val_"

        files = os.listdir(path_to_results)
        jsons_mask = np.array([bool(re.search(".json$", file)) for file in files])
        dataset_mask = np.array([bool(re.search(re_pattern, file)) for file in files])
        files_mask = jsons_mask & dataset_mask
        test_files = np.array(files)[files_mask]
        test_files = sorted(test_files)

        target_datasets = [
            "csfd",
            "csfd",
            "csfd",
            "mall",
            "mall",
            "mall",
            "facebook",
            "facebook",
            "facebook",
        ]

        anchor_size = [
            30,
            50,
            100,
            30,
            50,
            100,
            30,
            50,
            100,
        ]

        test_results_df = pd.DataFrame()
        for i, filename in enumerate(test_files):
            f = os.path.join(path_to_results, filename)

            df_ls = pd.read_json(f)
            df = pd.DataFrame(
                df_ls["target_" + target_datasets[i]].to_list(), index=df_ls.index
            )
            df = df.T
            df.index.name = "epoch"
            df = df.reset_index()
            df["dataset"] = target_datasets[i]
            df["anchor_size"] = anchor_size[i]

            test_results_df = pd.concat([test_results_df, df], axis=0)

        # save_path = os.path.join(path_to_results, dataset + ".csv")
        # test_results_df.to_csv(save_path)
    return test_results_df


if __name__ == "__main__":
    datasets = ["target_facebook", "target_mall", "target_csfd"]
    main(datasets)

    mall = pd.read_csv("output/train_info/finetuning/test/mall.csv", index_col=0)
    facebook = pd.read_csv(
        "output/train_info/finetuning/test/facebook.csv", index_col=0
    )
    csfd = pd.read_csv("output/train_info/finetuning/test/csfd.csv", index_col=0)
