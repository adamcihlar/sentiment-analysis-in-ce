import os
import pandas as pd
from src.config import paths
from src.reading.readers import (
    read_raw_sent,
    read_raw_responses,
    read_preprocessed_emails,
)
from loguru import logger
import itertools
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder


def get_df(new=False):
    """
    Load the dfs, merge them and save.
    Or just load the merged DataFrame.
    """
    if new:
        responses_df = read_preprocessed_emails()
        responses_df = responses_df[~responses_df.invalid]

        sent_df = read_raw_sent()

        ico_first_date = pd.DataFrame(sent_df.groupby("ico").min().date).reset_index()
        ico_first_date["join"] = ico_first_date.ico.astype(
            str
        ) + ico_first_date.date.astype(str)

        sent_df["join"] = sent_df.ico.astype(str) + sent_df.date.astype(str)

        merged = sent_df.merge(ico_first_date, how="left", on="join")
        merged["sent_first"] = False
        merged["sent_first"].loc[~merged.ico_y.isna()] = True

        sent_df["sent_first"] = merged["sent_first"]

        predictions_df = pd.read_csv("output/predictions/emails.csv", index_col=0)
        responses_df = responses_df.reset_index(drop=True)

        responses = predictions_df.merge(
            responses_df, how="inner", left_index=True, right_index=True
        )
        responses = responses[["y_pred", "id_2", "X"]]

        df = sent_df.merge(responses, how="inner", left_on="id", right_on="id_2")

        df.to_csv("data/final/emails_merged.csv")
    else:
        df = pd.read_csv("data/final/emails_merged.csv", index_col=0)
    return df


def get_regression_model_data(
    df,
    X_cols=["identity", "week"],
    y_col="y_pred",
    X_type=["discrete", "continuous"],
    sample_reduction="full",
    interactions=[("identity", "week")],
    constant=True,
):
    # filter the df
    if sample_reduction == "full":
        pass
    elif sample_reduction == "mul_answers_per_ico":
        icos = df.groupby(["ico"]).id_2.nunique() > 1
        sel_icos = icos.loc[icos].index
        df = df.loc[df.ico.isin(sel_icos)]
    else:
        logger.error("Invalid sample reduction")

    # prepare the df
    X = df[X_cols]
    y = df[y_col]

    assert len(X_cols) == len(X_type)

    cols = []
    col_names = []
    for i in range(len(X_cols)):
        if X_type[i] == "discrete":
            uniq = X[X_cols[i]].unique()[1:]
            if X_cols[i] == "identity":
                uniq = pd.Series(["A", "B", "C", "D"])
            for val in uniq:
                cols.append((X[X_cols[i]] == val) * 1)
                col_names.append(X_cols[i] + "_" + str(val))
        elif X_type[i] == "continuous":
            cols.append(X[X_cols[i]].astype(float))
            col_names.append(X_cols[i])
        else:
            logger.error("Invalid data type")
    X = pd.concat(cols, axis=1)
    X.columns = col_names

    for inter in interactions:
        cols_1 = [col for col in X if col.startswith(inter[0])]
        cols_2 = [col for col in X if col.startswith(inter[1])]

        new_cols = []
        new_cols_names = []
        for comb in itertools.product(cols_1, cols_2):
            new_cols.append(X[comb[0]] * X[comb[1]])
            new_cols_names.append("*".join(comb))
        interaction = pd.concat(new_cols, axis=1)
        interaction.columns = new_cols_names
        X = pd.concat([X, interaction], axis=1)

    if constant:
        X = sm.add_constant(X)

    return X, y


def model(X, y, **kvargs):
    mod = sm.OLS(y, X).fit(**kvargs)
    print(mod.summary())
    return mod


def main():
    # mul_answers_per_ico sample,  explain by identity, control week with cluster
    # errors
    df = get_df()
    X, y = get_regression_model_data(
        df,
        X_cols=["identity", "week"],
        y_col="y_pred",
        X_type=["discrete", "continuous"],
        sample_reduction="mul_answers_per_ico",
        interactions=[],
        constant=True,
    )
    wk = X.week
    X = X.iloc[:, :-1]
    mod = model(X, y, cov_type="cluster", cov_kwds={"groups": wk})
    print(mod.summary().as_latex())

    # mul_answers_per_ico sample,  explain by CZ x rest, control week with cluster
    # errors
    df = get_df()
    df["CZ"] = df.identity == "CZ"
    X, y = get_regression_model_data(
        df,
        X_cols=["CZ", "week"],
        y_col="y_pred",
        X_type=["discrete", "continuous"],
        sample_reduction="mul_answers_per_ico",
        interactions=[],
        constant=True,
    )
    wk = X.week
    X = X.iloc[:, :-1]
    mod = model(X, y, cov_type="cluster", cov_kwds={"groups": wk})
    print(mod.summary().as_latex())

    # mul_answers_per_ico sample,  explain by CZ x rest, control week with cluster
    # errors
    df = get_df()
    df["CZ"] = df.identity == "CZ"
    X, y = get_regression_model_data(
        df,
        X_cols=["CZ", "sent_first", "week"],
        y_col="y_pred",
        X_type=["discrete", "discrete", "continuous"],
        sample_reduction="mul_answers_per_ico",
        interactions=[],
        constant=True,
    )
    wk = X.week
    X = X.iloc[:, :-1]
    mod = model(X, y, cov_type="cluster", cov_kwds={"groups": wk})
    print(mod.summary().as_latex())

    # mul_answers_per_ico sample,  explain by identity, control continuous week
    # interaction week*identity
    df = get_df()
    X, y = get_regression_model_data(
        df,
        X_cols=["identity", "week"],
        y_col="y_pred",
        X_type=["discrete", "continuous"],
        sample_reduction="mul_answers_per_ico",
        interactions=[("identity", "week")],
        constant=True,
    )
    mod = model(X, y, cov_type="HC3")
    print(mod.summary().as_latex())

    # mul_answers_per_ico sample,  explain by CZ x rest, control continuous week
    # interaction week*identity
    df = get_df()
    df["CZ"] = df.identity == "CZ"
    X, y = get_regression_model_data(
        df,
        X_cols=["CZ", "week"],
        y_col="y_pred",
        X_type=["discrete", "continuous"],
        sample_reduction="mul_answers_per_ico",
        interactions=[("CZ", "week")],
        constant=True,
    )
    mod = model(X, y, cov_type="HC3")
    print(mod.summary().as_latex())
    pass


if __name__ == "__main__":
    ############ NO INTERACTIONS ############
    ### FULL SAMPLE
    # full sample,  explain by identity, control continuous week
    df = get_df()
    X, y = get_regression_model_data(
        df,
        X_cols=["identity", "week"],
        y_col="y_pred",
        X_type=["discrete", "continuous"],
        sample_reduction="full",
        interactions=[],
        constant=True,
    )
    model(X, y)

    # full sample,  explain by identity, control discrete week
    df = get_df()
    X, y = get_regression_model_data(
        df,
        X_cols=["identity", "week"],
        y_col="y_pred",
        X_type=["discrete", "discrete"],
        sample_reduction="full",
        interactions=[],
        constant=True,
    )
    model(X, y)

    # full sample,  explain by CZ x rest, control continuous week
    df = get_df()
    df["CZ"] = df.identity == "CZ"
    X, y = get_regression_model_data(
        df,
        X_cols=["CZ", "week"],
        y_col="y_pred",
        X_type=["discrete", "continuous"],
        sample_reduction="full",
        interactions=[],
        constant=True,
    )
    model(X, y)

    # full sample,  explain by CZ x rest, control discrete week
    df = get_df()
    df["CZ"] = df.identity == "CZ"
    X, y = get_regression_model_data(
        df,
        X_cols=["CZ", "week"],
        y_col="y_pred",
        X_type=["discrete", "discrete"],
        sample_reduction="full",
        interactions=[],
        constant=True,
    )
    model(X, y)

    ### LIMITED SAMPLE
    # mul_answers_per_ico sample,  explain by identity, control continuous week
    df = get_df()
    X, y = get_regression_model_data(
        df,
        X_cols=["identity", "week"],
        y_col="y_pred",
        X_type=["discrete", "continuous"],
        sample_reduction="mul_answers_per_ico",
        interactions=[],
        constant=True,
    )
    model(X, y)

    # mul_answers_per_ico sample,  explain by identity, control discrete week
    df = get_df()
    X, y = get_regression_model_data(
        df,
        X_cols=["identity", "week"],
        y_col="y_pred",
        X_type=["discrete", "discrete"],
        sample_reduction="mul_answers_per_ico",
        interactions=[],
        constant=True,
    )
    model(X, y)

    # mul_answers_per_ico sample,  explain by CZ x rest, control continuous week
    df = get_df()
    df["CZ"] = df.identity == "CZ"
    X, y = get_regression_model_data(
        df,
        X_cols=["CZ", "week"],
        y_col="y_pred",
        X_type=["discrete", "continuous"],
        sample_reduction="mul_answers_per_ico",
        interactions=[],
        constant=True,
    )
    model(X, y)
    model(X, y, cov_type="cluster", cov_kwds={"groups": X.week})

    # mul_answers_per_ico sample,  explain by CZ x rest, control discrete week
    df = get_df()
    df["CZ"] = df.identity == "CZ"
    X, y = get_regression_model_data(
        df,
        X_cols=["CZ", "week"],
        y_col="y_pred",
        X_type=["discrete", "discrete"],
        sample_reduction="mul_answers_per_ico",
        interactions=[],
        constant=True,
    )
    model(X, y)

    ############ INTERACTIONS ############
    ### FULL SAMPLE
    # full sample,  explain by identity, control continuous week
    # interaction week*identity
    df = get_df()
    X, y = get_regression_model_data(
        df,
        X_cols=["identity", "week"],
        y_col="y_pred",
        X_type=["discrete", "continuous"],
        sample_reduction="full",
        interactions=[("identity", "week")],
        constant=True,
    )
    model(X, y)

    # full sample,  explain by CZ x rest, control continuous week
    # interaction week*identity
    df = get_df()
    df["CZ"] = df.identity == "CZ"
    X, y = get_regression_model_data(
        df,
        X_cols=["CZ", "week"],
        y_col="y_pred",
        X_type=["discrete", "continuous"],
        sample_reduction="full",
        interactions=[("CZ", "week")],
        constant=True,
    )
    model(X, y, cov_type="HC3")

    ### LIMITED SAMPLE
    # mul_answers_per_ico sample,  explain by identity, control continuous week
    # interaction week*identity
    df = get_df()
    X, y = get_regression_model_data(
        df,
        X_cols=["identity", "week"],
        y_col="y_pred",
        X_type=["discrete", "continuous"],
        sample_reduction="mul_answers_per_ico",
        interactions=[("identity", "week")],
        constant=True,
    )
    model(X, y)

    # mul_answers_per_ico sample,  explain by CZ x rest, control continuous week
    # interaction week*identity
    df = get_df()
    df["CZ"] = df.identity == "CZ"
    X, y = get_regression_model_data(
        df,
        X_cols=["CZ", "week"],
        y_col="y_pred",
        X_type=["discrete", "continuous"],
        sample_reduction="mul_answers_per_ico",
        interactions=[("CZ", "week")],
        constant=True,
    )
    model(X, y, cov_type="HC3")

    df.text.value_counts()
    len(df.mail_text.value_counts())

    df.to_csv("output/emails_merged.csv")

    df.columns
