import os
import pandas as pd
import numpy as np
import json
import glob
import uuid


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


metadata_dir = "../../codex-use_cases/rareplanes/metadata"
filename = "RarePlanes_Metadata_Augmented_Processed_localized-tiled-controlpt.csv"
filename_binned = (
    "RarePlanes_Metadata_Augmented_Processed_localized-tiled-controlpt_Binned.csv"
)
metadata_dir_output = "../../codex-use_cases/rareplanesSIE/metadata_engineered/"

df = pd.read_csv(os.path.join(metadata_dir, filename))
df = df.drop([col for col in df.columns if "Unnamed" in col], axis=1)

####
"""df_path = os.path.join('../../codex-use_cases/iris/datasets/')# = pd.read_csv()
df = add_noise_feature(df_path, 0, 1, filename='iris_extended.csv')
print(df.head(n=10))

results = display_continuous_diagnostics(df, ["elevation", "sepal_length", "petal_length", "sepal_width", "petal_width"])
output_json_readable(results, print_json=True)"""

####
filename_reduced = (
    "RarePlanes_Metadata_Augmented_Processed_localized-tiled-controlpt-reducedbiome.csv"
)

"""df_low = lower_cc_one_feature(df, 'biome', 'Temperate Grasslands, Savannas & Shrublands')
df_low.to_csv(os.path.join(metadata_dir_output, filename_new), index=False)

###
filename_new = 'RarePlanes_Metadata_Augmented_Processed_localized-tiled-controlpt-full-.csv'
filename_new_train = 'RarePlanes_Metadata_Augmented_Processed_localized-tiled-controlpt-full-train-.csv'
filename_new_test = 'RarePlanes_Metadata_Augmented_Processed_localized-tiled-controlpt-full-test-.csv'

df_high = df.sample(8525)
n = int(len(df_low)/0.85)
df_high_train = df_high.sample(len(df_low))
df_high_test = df_high[~df_high['image_tile_id'].isin(df_high_train['image_tile_id'])].sample(int(0.15*n))

df_high_train.to_csv(os.path.join(metadata_dir_output, filename_new_train), index=False)
df_high_test.to_csv(os.path.join(metadata_dir_output, filename_new_test), index=False)
df_high.to_csv(os.path.join(metadata_dir_output, filename_new), index=False)"""

####
df_binned = pd.read_csv(os.path.join(metadata_dir, filename_binned))

pool3_binned, pool4_binned = grab_pools(df_binned)

n = int(len(df) / 2)

pool3 = df[df["image_tile_id"].isin(pool3_binned["image_tile_id"])]
pool4 = df[df["image_tile_id"].isin(pool4_binned["image_tile_id"])]

assert len(pool3) == len(pool3_binned)
assert len(pool4) == len(pool4_binned)

exit()
filename_new_pool3 = (
    "RarePlanes_Metadata_Augmented_Processed_localized-tiled-controlpt-pool3.csv"
)
pool3.to_csv(os.path.join(metadata_dir_output, filename_new_pool3))
filename_new_pool4 = (
    "RarePlanes_Metadata_Augmented_Processed_localized-tiled-controlpt-pool4.csv"
)
pool4.to_csv(os.path.join(metadata_dir_output, filename_new_pool4))

####
pool3 = pd.read_csv(os.path.join(metadata_dir_output, filename_new_pool3))
pool4 = pd.read_csv(os.path.join(metadata_dir_output, filename_new_pool4))
train3 = pool3.sample(int(0.85 * len(pool3)))
test3 = pool3[~pool3["image_tile_id"].isin(train3["image_tile_id"])]
train4 = pool4.sample(int(0.85 * len(pool4)))
test4 = pool4[~pool4["image_tile_id"].isin(train4["image_tile_id"])]
pool34 = pd.concat((pool3, pool4))

filename_pool_combined = "RarePlanes_Metadata_Augmented_Processed_localized-tiled-controlpt-pool3_pool4_combined.csv"
pool34.to_csv(os.path.join(metadata_dir_output, filename_pool_combined), index=False)

filename_train3 = (
    "RarePlanes_Metadata_Augmented_Processed_localized-tiled-controlpt-train3.csv"
)
filename_test3 = (
    "RarePlanes_Metadata_Augmented_Processed_localized-tiled-controlpt-test3.csv"
)
filename_train4 = (
    "RarePlanes_Metadata_Augmented_Processed_localized-tiled-controlpt-train4.csv"
)
filename_test4 = (
    "RarePlanes_Metadata_Augmented_Processed_localized-tiled-controlpt-test4.csv"
)

train3.to_csv(os.path.join(metadata_dir_output, filename_train3), index=False)
test3.to_csv(os.path.join(metadata_dir_output, filename_test3), index=False)
train4.to_csv(os.path.join(metadata_dir_output, filename_train4), index=False)
test4.to_csv(os.path.join(metadata_dir_output, filename_test4), index=False)


split34 = {
    "split_id": "split_34",
    "pool3": pool3["image_tile_id"].tolist(),
    "pool4": pool4["image_tile_id"].tolist(),
    "train3": train3["image_tile_id"].tolist(),
    "test3": test3["image_tile_id"].tolist(),
    "train4": train4["image_tile_id"].tolist(),
    "test4": test4["image_tile_id"].tolist(),
}
with open(
    os.path.join(
        "../../codex-use_cases/rareplanesSIE/splits-consolidated/",
        "split_pool3_pool4_combined.json",
    ),
    "w",
) as f:
    json.dump(split34, f)
