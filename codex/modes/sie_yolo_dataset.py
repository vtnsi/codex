import os

import typing_extensions
import importlib

importlib.reload(typing_extensions)


def __write_yml_cls(wf):
    wf.write("names:\n")
    wf.write("  0: Small Civil Transport/Utility\n")
    wf.write("  1: Medium Civil Transport/Utility\n")
    wf.write("  2: Large Civil Transport/Utility\n")
    wf.write("  3: Military Transport/Utility/AWAC\n")
    wf.write("  4: Military Bomber\n")
    wf.write("  5: Military Fighter/Interceptor/Attack\n")
    wf.write("  6: Military Trainer\n")


def __SIE_split_txt(
    SIE_id,
    data_dir,
    dataset_dir,
    img_ids,
    subset_name,
    overwrite=False,
    ctrl=False,
    dataset_name="rareplanes_SIE",
):
    image_repository_path = os.path.join(data_dir, "images")
    label_repository_path = os.path.join(data_dir, "labels")

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    with open(
        os.path.join(dataset_dir, "{}{}.txt".format(SIE_id, subset_name)), "w"
    ) as f:
        for img in img_ids:
            f.write(
                str(os.path.join(image_repository_path, "{}.png{}".format(img, "\n")))
            )


def __SIE_yaml_txt(SIE_id, dataset_dir, inex):
    if "x" in SIE_id:
        return
    with open(
        os.path.join(dataset_dir, "{}{}_data.yml".format(SIE_id, inex)), "w"
    ) as wf:
        wf.write("train: {}".format(os.path.join("{}train.txt\n".format(SIE_id))))
        wf.write("val: {}".format(os.path.join("{}val.txt\n".format(SIE_id))))
        wf.write(
            "test: {}".format(os.path.join("{}test_{}.txt\n".format(SIE_id, inex)))
        )

        wf.write("\n")
        __write_yml_cls(wf)


def dataset_assembly(
    SIE_id,
    data_dir,
    dataset_dir,
    train_ids,
    val_ids,
    test_ids,
    test_ex_ids=None,
    overwrite=True,
    mode_text=True,
):
    if mode_text:
        print("Distributing data...")
        __SIE_split_txt(
            SIE_id, data_dir, dataset_dir, train_ids, "train", overwrite=overwrite
        )
        __SIE_split_txt(
            SIE_id, data_dir, dataset_dir, val_ids, "val", overwrite=overwrite
        )
        __SIE_split_txt(
            SIE_id, data_dir, dataset_dir, test_ids, "test_incl", overwrite=overwrite
        )
        __SIE_split_txt(
            SIE_id, data_dir, dataset_dir, test_ex_ids, "test_excl", overwrite=overwrite
        )
        __SIE_yaml_txt(SIE_id, dataset_dir, "incl")
        __SIE_yaml_txt(SIE_id, dataset_dir, "excl")


def data_distribution_SIE(SIE_splits, data_dir, dataset_dir, overwrite, mode_text):
    for SIE_id in SIE_splits:
        print(SIE_id)
        SIE_splits[SIE_id]

        train_id = SIE_splits[SIE_id]["train"]
        val_id = SIE_splits[SIE_id]["validation"]
        test_covered_id = SIE_splits[SIE_id]["test_included"]
        test_withheld_id = SIE_splits[SIE_id]["test_excluded"]

        print(
            "TRAIN, VAL, TEST IN/EX\n",
            len(train_id),
            len(val_id),
            len(test_covered_id),
            len(test_withheld_id),
        )
        dataset_assembly(
            SIE_id,
            data_dir,
            dataset_dir,
            train_id,
            val_id,
            test_covered_id,
            test_withheld_id,
            overwrite=overwrite,
            mode_text=mode_text,
        )

    return


def __write_yml_cls(wf):
    wf.write("names:\n")
    wf.write("  0: Small Civil Transport/Utility\n")
    wf.write("  1: Medium Civil Transport/Utility\n")
    wf.write("  2: Large Civil Transport/Utility\n")
    wf.write("  3: Military Transport/Utility/AWAC\n")
    wf.write("  4: Military Bomber\n")
    wf.write("  5: Military Fighter/Interceptor/Attack\n")
    wf.write("  6: Military Trainer\n")


def __SIE_split_txt(
    SIE_id,
    data_dir,
    dataset_dir,
    img_ids,
    subset_name,
    overwrite=False,
    ctrl=False,
    dataset_name="rareplanes_SIE",
):
    image_repository_path = os.path.join(data_dir, "images")
    label_repository_path = os.path.join(data_dir, "labels")

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    with open(
        os.path.join(dataset_dir, "{}{}.txt".format(SIE_id, subset_name)), "w"
    ) as f:
        for img in img_ids:
            f.write(
                str(os.path.join(image_repository_path, "{}.png{}".format(img, "\n")))
            )


def __SIE_yaml_txt(SIE_id, dataset_dir, inex):
    if "x" in SIE_id:
        return
    with open(
        os.path.join(dataset_dir, "{}{}_data.yml".format(SIE_id, inex)), "w"
    ) as wf:
        wf.write("train: {}".format(os.path.join("{}train.txt\n".format(SIE_id))))
        wf.write("val: {}".format(os.path.join("{}val.txt\n".format(SIE_id))))
        wf.write(
            "test: {}".format(os.path.join("{}test_{}.txt\n".format(SIE_id, inex)))
        )

        wf.write("\n")
        __write_yml_cls(wf)


def dataset_assembly(
    SIE_id,
    data_dir,
    dataset_dir,
    train_ids,
    val_ids,
    test_ids,
    test_ex_ids=None,
    overwrite=True,
    mode_text=True,
):
    if mode_text:
        print("Distributing data...")
        __SIE_split_txt(
            SIE_id, data_dir, dataset_dir, train_ids, "train", overwrite=overwrite
        )
        __SIE_split_txt(
            SIE_id, data_dir, dataset_dir, val_ids, "val", overwrite=overwrite
        )
        __SIE_split_txt(
            SIE_id, data_dir, dataset_dir, test_ids, "test_incl", overwrite=overwrite
        )
        __SIE_split_txt(
            SIE_id, data_dir, dataset_dir, test_ex_ids, "test_excl", overwrite=overwrite
        )
        __SIE_yaml_txt(SIE_id, dataset_dir, "incl")
        __SIE_yaml_txt(SIE_id, dataset_dir, "excl")


def data_distribution_SIE(SIE_splits, data_dir, dataset_dir, overwrite, mode_text):
    for SIE_id in SIE_splits:
        print(SIE_id)
        SIE_splits[SIE_id]

        train_id = SIE_splits[SIE_id]["train"]
        val_id = SIE_splits[SIE_id]["validation"]
        test_covered_id = SIE_splits[SIE_id]["test_included"]
        test_withheld_id = SIE_splits[SIE_id]["test_excluded"]

        print(
            "TRAIN, VAL, TEST IN/EX\n",
            len(train_id),
            len(val_id),
            len(test_covered_id),
            len(test_withheld_id),
        )
        dataset_assembly(
            SIE_id,
            data_dir,
            dataset_dir,
            train_id,
            val_id,
            test_covered_id,
            test_withheld_id,
            overwrite=overwrite,
            mode_text=mode_text,
        )

    return


def get_image_paths(interaction_id, dataset_dir):
    with open(os.path.join(dataset_dir, interaction_id + "test_incl.txt")) as f:
        test_in_img_paths = list(f)
    test_in_img_paths = [path.split("\n")[0] for path in test_in_img_paths]

    if interaction_id != "_x_":
        with open(os.path.join(dataset_dir, interaction_id + "test_excl.txt")) as f:
            test_ex_img_paths = list(f)
        test_ex_img_paths = [path.split("\n")[0] for path in test_ex_img_paths]
    else:
        test_ex_img_paths = []

    return test_in_img_paths, test_ex_img_paths
