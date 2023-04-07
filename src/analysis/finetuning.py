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
        path_to_results = os.path.join(paths.OUTPUT_INFO_FINETUNING, "test")
        re_pattern = "^" + dataset

        files = os.listdir(path_to_results)
        jsons_mask = np.array([bool(re.search(".json$", file)) for file in files])
        dataset_mask = np.array([bool(re.search(re_pattern, file)) for file in files])
        files_mask = jsons_mask & dataset_mask
        test_files = np.array(files)[files_mask]
        test_files = sorted(test_files)

        encoders = [
            "independent",
            "independent",
            "shared",
            "shared",
            "shared",
            "shared",
            "shared",
            "shared",
            "shared",
        ]
        classifiers = [
            "independent",
            "independent",
            "independent",
            "independent",
            "independent",
            "independent",
            "shared",
            "shared",
            "shared",
        ]

        task = [
            "ordinal",
            "multiclass",
            "ordinal",
            "multiclass",
            "ordinal",
            "ordinal",
            "ordinal",
            "ordinal",
            "ordinal",
        ]

        test_results_df = pd.DataFrame()
        for i, filename in enumerate(test_files):
            f = os.path.join(path_to_results, filename)

            df = (
                pd.read_json(f)
                .loc[["f1-score", "mae", "rmse"]]
                .dropna(axis=1)
                .drop(columns="accuracy")
            )
            cols = list(itertools.product(df.index, df.columns))
            col_names = [col[0] + "_" + col[1] for col in cols]
            vals = df.values.flatten()
            df = pd.Series(vals, index=col_names)

            df["model"] = filename
            df["encoder"] = encoders[i]
            df["classifier"] = classifiers[i]
            df["task"] = task[i]
            test_results_df = pd.concat([test_results_df, df.to_frame().T], axis=0)

        save_path = os.path.join(path_to_results, dataset + ".csv")
        test_results_df.to_csv(save_path)
    pass


if __name__ == "__main__":
    datasets = ["facebook", "mall", "csfd"]
    main(datasets)
