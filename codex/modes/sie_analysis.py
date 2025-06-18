import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


def set_box_color(fig_bp, color):
    plt.setp(fig_bp["boxes"], color=color)
    plt.setp(fig_bp["whiskers"], color=color)
    plt.setp(fig_bp["caps"], color=color)
    plt.setp(fig_bp["medians"], color=color)


def box_plot_SIE(df):
    excluded_features = df["Feature_Excluded"].unique()
    excluded_features.sort()
    print(excluded_features)
    perf_in = {
        feature: df[df["Feature_Excluded"] == feature][
            df["Included_in_Training"] == "Yes"
        ]["Precision"].tolist()
        for feature in excluded_features
    }
    perf_ex = {
        feature: df[df["Feature_Excluded"] == feature][
            df["Included_in_Training"] == "No"
        ]["Precision"].tolist()
        for feature in excluded_features
    }

    in_p = [perf_in[feature] for feature in perf_in]
    ex_p = [perf_ex[feature] for feature in perf_ex]

    bp_in = plt.boxplot(
        in_p, positions=np.array(range(len(in_p))) * 2 - 0.4, sym="", widths=0.6
    )
    bp_ex = plt.boxplot(
        ex_p, positions=np.array(range(len(ex_p))) * 2 + 0.4, sym="", widths=0.6
    )
    # SEt box parameters
    set_box_color(bp_in, "blue")
    set_box_color(bp_ex, "red")
    print(perf_in["biome"])

    plt.savefig("./test-a.png")
    return


def preprocess_binreg(df: pd.DataFrame, metrics: list):
    df_proc = df[
        ["Feature Excluded", "Included in Training", "Test Set Size"] + metrics
    ]
    df_proc = df_proc.rename(
        {
            "Feature Excluded": "FE",
            "Included in Training": "IT",
            "Test Set Size": "Freq",
        },
        axis=1,
    )
    df_proc = df_proc.drop(df_proc[df_proc["FE"] == "Control"].index, axis=0).drop(
        df_proc[df_proc["FE"] == "Baseline"].index, axis=0
    )

    features_excluded = df_proc["FE"].unique().tolist()
    features_excluded.sort()

    for feature in features_excluded:
        # num_levels = len(df_proc[df_proc['FE'] == feature]['Level Excluded'].unique())
        # levels = df_proc[df_proc['FE'] == feature]['Level Excluded'].unique()

        df_proc.insert(
            len(df_proc.columns.tolist()),
            "{}_No".format(feature),
            np.zeros(len(df_proc), dtype=int),
        )
        df_proc.insert(
            len(df_proc.columns.tolist()),
            "{}_Yes".format(feature),
            np.zeros(len(df_proc), dtype=int),
        )

        for i in range(len(df_proc)):
            row = df_proc.iloc[i]
            if row["FE"] == feature and row["IT"] == "Yes":
                df_proc.at[i, "{}_Yes".format(feature)] = 1
            elif row["FE"] == feature and row["IT"] == "No":
                df_proc.at[i, "{}_No".format(feature)] = 1
            else:
                continue
                raise ValueError(
                    "Anomaly in dataframe for feature {}, sample {}".format(feature, i)
                )

    df_proc = df_proc.drop(["FE", "IT"], axis=1)
    # print(df_proc.head())
    return df_proc, features_excluded


def SIE_binomial_regression_main(
    df: pd.DataFrame, metrics: list, selected_metric, features: list
):
    # features = [f for f in features if f != 'CONTROL']
    df_proc, features_excluded = preprocess_binreg(df, metrics)
    data_matrix = df_proc.iloc[:, 4:]

    contrasts = [
        [0 for j in range(2 * len(features_excluded))] for feature in features_excluded
    ]
    contrast_names = []
    # print(contrasts)
    for i, contrast_row in enumerate(contrasts):
        contrast_row[2 * i] = 1
        contrast_row[(2 * i) + 1] = -1

    for feature in features_excluded:
        contrast_names.append("{}_Yes-{}_No".format(feature, feature))

    contrasts_hc = [
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1],
    ]
    assert contrasts_hc == contrasts

    model = sm.GLM(
        endog=df_proc[selected_metric],
        exog=data_matrix,
        family=sm.families.Binomial(),
        n_trials=np.asarray(df_proc["Freq"]),
    ).fit()
    contrasts_model = model.t_test(contrasts)
    print("------ BINOMIAL REGRESSION TESTING FOR {} ------".format(selected_metric))
    print(model.summary())
    print(contrasts_model.summary())

    return model.summary(), contrasts_model.summary(), contrast_names
