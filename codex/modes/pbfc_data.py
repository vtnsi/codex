import pandas as pd
import json
import glob
import os
import sklearn
import sklearn.model_selection
import sklearn.preprocessing


def output_json_readable(
    json_obj: dict,
    print_json=False,
    write_json=False,
    file_path="",
    sort=False,
    truncate_lists=False,
):
    """
    Formats JSON object to human-readable format with print/save options.
    """
    if truncate_lists:
        for key in json_obj:
            if type(json_obj[key]) is list:
                json_obj[key] = json_obj[key][:5]
                json_obj[key].append("...")

    json_str = json.dumps(json_obj, sort_keys=sort, indent=4, separators=(",", ": "))

    if print_json:
        print(json_str)

    if write_json:
        if file_path == "":
            file_path = "output_0{}.json".format(len(glob.glob("output_0*.json")))
        with open(file_path, "w") as f:
            f.write(json_str)

    return json_obj


def one_hot_df(df_original, encoded_features_list):
    df_encoded = df_original  # .copy(deep=True)
    for feature in encoded_features_list:
        one_hot = pd.get_dummies(df_original[feature], prefix=feature, dtype="int")
        df_encoded = pd.concat([df_encoded, one_hot], axis=1)
    df_encoded = df_encoded.drop(encoded_features_list, axis=1)
    return df_encoded


def prep_split_data(
    df: pd.DataFrame,
    target_col: str,
    id_col: str,
    split_dir,
    name,
    drop_list=[],
    random=False,
    **kwargs,
):
    train_df = kwargs["train_df"]
    test_df = kwargs["test_df"]

    X_train, Y_train, train_ids = prep_data(train_df, target_col, id_col, drop_list)
    X_test, Y_test, test_ids = prep_data(test_df, target_col, id_col, drop_list)

    split, split_filename = write_split(split_dir, name, train_ids, test_ids)
    assert (
        id_col not in X_train.columns.tolist()
        and target_col not in X_train.columns.tolist()
    )

    return X_train, X_test, Y_train, Y_test, split_filename


def prep_split_data_retain(
    df: pd.DataFrame,
    target_col: str,
    id_col: str,
    split_dir,
    name,
    drop_list=[],
    random=False,
    **kwargs,
):
    train_df = kwargs["train_df"]
    test_df = kwargs["test_df"]

    X_train, Y_train, train_ids = prep_data(train_df, target_col, id_col, drop_list)
    X_test, Y_test, test_ids = prep_data(test_df, target_col, id_col, drop_list)

    split, split_filename = write_split(split_dir, name, train_ids, test_ids)
    assert (
        id_col not in X_train.columns.tolist()
        and target_col not in X_train.columns.tolist()
    )

    return X_train, X_test, Y_train, Y_test, split_filename


def prep_data(
    df: pd.DataFrame,
    target_col: str,
    id_col: str,
    drop_list: list = [],
    preserve_ids=False,
):
    if preserve_ids:
        drop_list.append(target_col)
    else:
        drop_list.append(target_col)
        drop_list.append(id_col)

    try:
        ids = [str(id) for id in df[id_col].tolist()]
    except KeyError:
        print("Assuming index column is already named for {}".format(id_col))
        ids = [str(id) for id in df.index.tolist()]

    X = df.drop(drop_list, axis=1)
    Y = df[target_col]

    return X, Y, ids


def write_split(
    split_dir,
    name,
    train_ids,
    test_ids,
):
    split = {"split_id": name, "train": train_ids, "test": test_ids}

    # pd.concat([X_train, Y_train], axis=1).to_csv(os.path.join(data_dir, '{}_train.csv'.format(experiment_name)), index=True)
    # pd.concat([X_test, Y_test], axis=1).to_csv(os.path.join(data_dir, '{}_test.csv'.format(experiment_name)), index=True)

    split_filename = "split_{}.json".format(name)
    output_json_readable(
        split, write_json=True, file_path=os.path.join(split_dir, split_filename)
    )

    return split, split_filename


def format_df_nd_all(
    X_df: pd.DataFrame,
    Y_ser: pd.Series,
    scaler: sklearn.preprocessing.StandardScaler = None,
):
    if scaler is None:
        X_nd = X_df.values
    else:
        X_nd = scaler.transform(X_df.values)
        print("SCALING DATA")

    Y_nd = Y_ser.values
    print("X, Y dim:", X_nd.shape, Y_nd.shape)
    print(X_nd[:5])
    return X_nd, Y_nd, scaler
