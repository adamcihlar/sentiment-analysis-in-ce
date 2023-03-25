import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go

from src.config.adaptation_schedules import Mall
from src.config.adaptation_schedules import CSFD
from src.config.adaptation_schedules import Facebook
from src.config import paths


def get_df_for_plot(target, target_schedule, path_to_results):
    """
    Loads all training information from adaptation for a given target and
    combines the most important metrics into one df.
    """
    # information about models
    models = ["_".join(mod) for mod in target_schedule.models]
    source_ds = [mod[2] for mod in target_schedule.models]

    encoders = ["independent", "independent", "shared", "shared", "shared"]
    classifiers = ["independent", "independent", "independent", "independent", "shared"]

    # prepare variables for the axis
    evaluation_df = pd.DataFrame(
        {
            "model": models,
            "encoders": encoders,
            "classifiers": classifiers,
            "source_dataset": source_ds,
        }
    )

    # results from training
    files = os.listdir(path_to_results)
    files_mask = np.array([bool(re.search("test", file)) for file in files])
    test_files = np.array(files)[files_mask]
    test_files = sorted(test_files)

    test_results_df = pd.DataFrame()
    for filename in test_files:
        f = os.path.join(path_to_results, filename)
        df = pd.read_json(f).rename_axis("epoch").reset_index()
        df = df.loc[df.epoch == 0]
        test_results_df = pd.concat([test_results_df, df], axis=0)
    div = int(len(test_results_df) / len(models))
    test_results_df = test_results_df.iloc[::div, :]

    df = pd.concat([evaluation_df, test_results_df.reset_index(drop=True)], axis=1)

    # create dummies for parallel_coordinates plot
    df_ext = pd.DataFrame({"encoders": df["encoders"].unique()})
    df_ext["encoders_d"] = df_ext.index
    df = pd.merge(df, df_ext, on="encoders", how="left")

    df_ext = pd.DataFrame({"classifiers": df["classifiers"].unique()})
    df_ext["classifiers_d"] = df_ext.index
    df = pd.merge(df, df_ext, on="classifiers", how="left")

    df_ext = pd.DataFrame({"source_dataset": df["source_dataset"].unique()})
    df_ext["source_dataset_d"] = df_ext.index
    df = pd.merge(df, df_ext, on="source_dataset", how="left")

    gs_path = os.path.join(paths.OUTPUT_INFO_FINETUNING, "test", target + ".csv")
    golden = pd.read_csv(gs_path, index_col=0)
    micro_golden = golden.iloc[np.argmax(golden["macro avg"])].accuracy
    macro_golden = np.max(golden["macro avg"])
    weighted_golden = golden.iloc[np.argmax(golden["macro avg"])]["weighted avg"]

    df["micro_drop"] = df["micro"] - micro_golden
    df["macro_drop"] = df["macro"] - macro_golden
    df["weighted_drop"] = df["weighted"] - weighted_golden
    return df


def get_ticks(minimum, maximum, num_ticks=5):
    step = ((maximum - minimum) / (num_ticks - 1)) + 0.000000001
    ticks = np.arange(minimum, maximum + step, step)
    return ticks


def plot_results(df, target):
    """
    Saves parallel coordinates plot for analysis of the adaptation.
    """
    dimensions = list(
        [
            dict(
                label="Encoder",
                range=[0, df.encoders_d.max()],
                tickvals=df.encoders_d.unique(),
                ticktext=df.encoders.unique(),
                values=df.encoders_d,
            ),
            dict(
                label="Classifier",
                range=[0, df.classifiers_d.max()],
                tickvals=df.classifiers_d.unique(),
                ticktext=df.classifiers.unique(),
                values=df.classifiers_d,
            ),
            dict(
                label="Source dataset",
                range=[0, df.source_dataset_d.max()],
                tickvals=df.source_dataset_d.unique(),
                ticktext=df.source_dataset.unique(),
                values=df.source_dataset_d,
            ),
            dict(
                label="F1-micro",
                range=[df.micro.min(), df.micro.max()],
                # range=[df.micro.min() + 0.2, df.micro.max()], # mall adjust
                tickvals=get_ticks(df.micro.min(), df.micro.max()),
                # tickvals=get_ticks(df.micro.min() + 0.2, df.micro.max()), # mall adjust
                ticktext=get_ticks(df.micro.min(), df.micro.max()),
                # ticktext=get_ticks(df.micro.min() + 0.2, df.micro.max()), # mall adjust
                values=df.micro,
            ),
            dict(
                label="F1-macro",
                range=[df.macro.min(), df.macro.max()],
                tickvals=get_ticks(df.macro.min(), df.macro.max()),
                ticktext=get_ticks(df.macro.min(), df.macro.max()),
                values=df.macro,
            ),
            dict(
                label="F1-weighted",
                range=[df.weighted.min(), df.weighted.max()],
                # range=[df.weighted.min() + 0.2, df.weighted.max()], # mall adjust
                tickvals=get_ticks(df.weighted.min(), df.weighted.max()),
                # tickvals=get_ticks(df.weighted.min() + 0.2, df.weighted.max()), # mall adjust
                ticktext=get_ticks(df.weighted.min(), df.weighted.max()),
                # ticktext=get_ticks(df.weighted.min() + 0.2, df.weighted.max()), # mall adjust
                values=df.weighted,
            ),
            dict(
                label="F1-micro drop",
                range=[df.micro_drop.min(), df.micro_drop.max()],
                tickvals=get_ticks(df.micro_drop.min(), df.micro_drop.max()),
                ticktext=get_ticks(df.micro_drop.min(), df.micro_drop.max()),
                values=df.micro_drop,
            ),
            dict(
                label="F1-macro drop",
                range=[df.macro_drop.min(), df.macro_drop.max()],
                tickvals=get_ticks(df.macro_drop.min(), df.macro_drop.max()),
                ticktext=get_ticks(df.macro_drop.min(), df.macro_drop.max()),
                values=df.macro_drop,
            ),
            dict(
                label="F1-weighted drop",
                range=[df.weighted_drop.min(), df.weighted_drop.max()],
                tickvals=get_ticks(df.weighted_drop.min(), df.weighted_drop.max()),
                ticktext=get_ticks(df.weighted_drop.min(), df.weighted_drop.max()),
                values=df.weighted_drop,
            ),
            dict(
                label="Test loss",
                range=[df.test_loss.min(), df.test_loss.max()],
                tickvals=get_ticks(df.test_loss.min(), df.test_loss.max()),
                ticktext=get_ticks(df.test_loss.min(), df.test_loss.max()),
                values=df.test_loss,
            ),
        ]
    )

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=df["epoch"],
            ),
            dimensions=dimensions,
        )
    )
    file_name = "par_cord_" + target + ".html"
    save_path = os.path.join(paths.OUTPUT_ASSETS, "domain_transfer", file_name)
    fig.write_html(save_path)
    pass


if __name__ == "__main__":
    # mall
    target = "mall"
    target_schedule = Mall
    path_to_results = paths.OUTPUT_INFO_ADAPTATION_MALL

    df = get_df_for_plot(target, target_schedule, path_to_results)
    plot_results(df, target)

    # csfd
    target = "csfd"
    target_schedule = CSFD
    path_to_results = paths.OUTPUT_INFO_ADAPTATION_CSFD

    df = get_df_for_plot(target, target_schedule, path_to_results)
    plot_results(df, target)

    # facebook
    target = "facebook"
    target_schedule = Facebook
    path_to_results = paths.OUTPUT_INFO_ADAPTATION_FACEBOOK

    df = get_df_for_plot(target, target_schedule, path_to_results)
    plot_results(df, target)


# validation sets - do I care?
files_mask = np.array([bool(re.search("val", file)) for file in files])
val_files = np.array(files)[files_mask]
val_files = sorted(val_files)
