import os
import pandas as pd
import numpy as np
import glob
import json
import uuid

def reorder_df_byindex(df_to_sort, order_ascending=True):
    if type(df_to_sort) is dict:
        print("Sorting and transposing performance JSON...")
        df_to_sort = pd.DataFrame(df_to_sort).sort_index().transpose()
    assert type(df_to_sort) is pd.DataFrame
    df_to_sort = df_to_sort.sort_index(ascending=order_ascending)
    return df_to_sort


def reorder_df_by_sample(
    sample_id_col: str, df_to_sort: pd.DataFrame, order_ascending=True
):
    df_to_sort = df_to_sort.sort_values(by=[sample_id_col], ascending=order_ascending)
    df_to_sort = df_to_sort.reset_index().drop(columns="index")
    return df_to_sort


def df_slice_by_id_reorder(
    sample_id_col: str, dataset_df: pd.DataFrame, sample_ids: list, order_ascending=True
):
    sliced_df = dataset_df.loc[dataset_df[sample_id_col].isin(sample_ids)]
    sliced_df = (
        sliced_df.sort_values(by=sample_id_col, ascending=order_ascending)
        .reset_index()
        .drop(columns="index")
    )
    return sliced_df

def add_noise_feature(df_dir, high, low, filename=None, save=True):
    if type(df_dir) is str:
        df = pd.read_csv(os.path.join(df_dir, filename))

    n = len(df)

    ctrl = pd.Series(np.random.uniform(high, low, n))
    df.insert(len(df.columns), "CONTROL", ctrl)

    if save and type(df_dir) is str:
        df.to_csv(os.path.join(df_dir, "{}_ctrl.csv".format(filename.split(".")[0])))

    return df


def display_continuous_diagnostics(df: pd.DataFrame, feature_list):
    results_dict = {}
    for feature in feature_list:
        feature_col = df[feature]
        results_dict[feature] = {
            "max": np.max(feature_col),
            "min": np.min(feature_col),
            "avg": np.mean(feature_col),
            "med": np.median(feature_col),
        }
    return results_dict


def output_json_readable(
    json_obj: dict, print_json=False, write_json=False, file_path="", sort=False
):
    """
    Formats JSON object to human-readable format with print/save options.
    """
    json_str = json.dumps(json_obj, sort_keys=sort, indent=4, separators=(",", ": "))

    if print_json:
        print(json_str)

    if write_json:
        if file_path == "":
            file_path = "output_0{}.json".format(len(glob.glob("output_0*.json")))
        with open(file_path, "w") as f:
            f.write(json_str)

    return json_obj


def lower_cc(data_dir, data_file):
    df = pd.read_csv(os.path.join(data_dir, data_file))
    # df = pd.read_csv(os.path.join(metadata_dir, metadata_file)).drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)

    df_new = pd.DataFrame()

    for row in df.iterrows():
        df_new = df_new._append(df.iloc[0])

    for i in range(1, 6):
        df_new = df_new._append(df.iloc[i])

    return df_new


def lower_cc_one_feature(df, feature, level):
    df_new = df[df[feature] == level]

    return df_new


def grab_pools(df_binned):
    levels_hour_of_day = df_binned["Hour_of_Day"].unique()
    levels_avg_pan_res = df_binned["avg_pan_resolution"].unique()
    import random

    random.shuffle(levels_hour_of_day)
    random.shuffle(levels_avg_pan_res)

    pool3 = df_binned.loc[
        df_binned["Hour_of_Day"].isin([levels_hour_of_day[0]])
        & df_binned["avg_pan_resolution"].isin([levels_avg_pan_res[0]])
        | df_binned["Hour_of_Day"].isin([levels_hour_of_day[0]])
        & df_binned["avg_pan_resolution"].isin([levels_avg_pan_res[1]])
        | df_binned["Hour_of_Day"].isin([levels_hour_of_day[0]])
        & df_binned["avg_pan_resolution"].isin([levels_avg_pan_res[2]])
    ]

    pool4 = df_binned.loc[
        df_binned["Hour_of_Day"].isin([levels_hour_of_day[2]])
        & df_binned["avg_pan_resolution"].isin([levels_avg_pan_res[0]])
        | df_binned["Hour_of_Day"].isin([levels_hour_of_day[2]])
        & df_binned["avg_pan_resolution"].isin([levels_avg_pan_res[1]])
        | df_binned["Hour_of_Day"].isin([levels_hour_of_day[2]])
        & df_binned["avg_pan_resolution"].isin([levels_avg_pan_res[2]])
    ]

    print(pool3["image_tile_id"].isin(pool4["image_tile_id"]).unique())
    if (
        len(pool3["avg_pan_resolution"].unique()) != 3
        or len(pool4["avg_pan_resolution"].unique()) != 3
    ):
        print("Combintorial gap. Running again.")
        grab_pools(df_binned)

    max_len = np.min([len(pool3), len(pool4)])
    pool3 = pool3.sample(max_len)
    pool4 = pool4.sample(max_len)
    print(len(pool4), len(pool3))
    return pool3, pool4


def uuider(abstract_df, codex_directory):
    for i in range(len(abstract_df)):
        abstract_df.loc[i, "id"] = uuid.uuid4(i)
        abstract_df = abstract_df.sample(len(abstract_df))
        abstract_df.to_csv(
            os.path.join(codex_directory, "data", "abstract_native.csv"), index=False
        )