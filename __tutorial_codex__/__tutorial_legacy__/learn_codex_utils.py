import directory_tree
import pandas as pd
import os
from IPython.display import display


def describe_feature_list(df: pd.DataFrame, chosen_features: list[str]):
    for feature in chosen_features:
        try:
            col = df[feature]

        except:
            raise KeyError(
                f"Feature {feature} does not exist in dataset. Please re-examine feature list."
            )

        if col.dtype is object:
            print(f"Categorical feature {feature} ---------------")
            display(col.value_counts())
        else:
            print(f"Continuous feature {feature} ---------------")
            display(col.describe())

    return


def write_binning_file(
    df: pd.DataFrame, chosen_features: list[str], feature_lines: list[str], codex_dir
):
    for line in feature_lines:
        if line == "":
            feature_lines.remove(line)
            continue
        feature = line.split(": ")[0]
        values = line.split(": ")[1].split(";")

        if feature not in df.columns:
            raise KeyError(f"Feature name {feature} not in dataset!")
        elif feature not in chosen_features:
            feature_lines.remove(line)

    bin_path = os.path.join(codex_dir, "binning", "binning_abstract_custom.txt")
    with open(bin_path, "w") as f:
        for i, line in enumerate(feature_lines):
            f.write(line)
            if i != len(feature_lines) - 1:
                f.write("\n")

    print("BINNING FILE WRITTEN UNDER {}".format(bin_path))
    return bin_path


def display_directory(codex_dir):
    print(directory_tree.DisplayTree(codex_dir))
