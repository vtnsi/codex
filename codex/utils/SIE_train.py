import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torchvision
from multiprocessing import Process, Pool

from ultralytics import YOLO
from ultralytics.utils.plotting import plt_color_scatter
from ultralytics.utils.plotting import plot_results

import typing_extensions
import importlib
importlib.reload(typing_extensions)
'''print("PyTorch ver: ", torch.__version__)
print("TorchVision ver: ", torchvision.__version__)'''


def write_yml_cls(wf):
    wf.write("names:\n")
    wf.write("  0: Small Civil Transport/Utility\n")
    wf.write("  1: Medium Civil Transport/Utility\n")
    wf.write("  2: Large Civil Transport/Utility\n")
    wf.write("  3: Military Transport/Utility/AWAC\n")
    wf.write("  4: Military Bomber\n")
    wf.write("  5: Military Fighter/Interceptor/Attack\n")
    wf.write("  6: Military Trainer\n")


def SIE_split_txt(
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


def SIE_yaml_txt(SIE_id, dataset_dir, inex):
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
        write_yml_cls(wf)


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
        SIE_split_txt(
            SIE_id, data_dir, dataset_dir, train_ids, "train", overwrite=overwrite
        )
        SIE_split_txt(
            SIE_id, data_dir, dataset_dir, val_ids, "val", overwrite=overwrite
        )
        SIE_split_txt(
            SIE_id, data_dir, dataset_dir, test_ids, "test_incl", overwrite=overwrite
        )
        SIE_split_txt(
            SIE_id, data_dir, dataset_dir, test_ex_ids, "test_excl", overwrite=overwrite
        )
        SIE_yaml_txt(SIE_id, dataset_dir, "incl")
        SIE_yaml_txt(SIE_id, dataset_dir, "excl")


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


def train_SIE(
    SIE_splits,
    dataset_dir,
    training_dir,
    epochs=300,
    batch=256,
    mode_text=True,
    devices=None,
    force_resume=True,
):
    pbar = tqdm(SIE_splits, leave=False)
    for SIE_id in SIE_splits:
        exists = os.path.exists(os.path.join(training_dir, "train_wo{}".format(SIE_id)))
        if exists:
            continue
            with open(
                os.path.join(training_dir, "train_wo{}".format(SIE_id), "results.csv")
            ) as f:
                resume = (
                    len(
                        pd.read_csv(
                            os.path.join(
                                training_dir, "train_wo{}".format(SIE_id), "results.csv"
                            )
                        )
                    )
                    == epochs
                )

        resume = False

        dataset_file = os.path.join(dataset_dir, "{}incl_data.yml".format(SIE_id))
        model_full_train = train(
            SIE_id,
            training_dir,
            dataset_file,
            epochs=epochs,
            batch=batch,
            mode_text=mode_text,
            devices=devices,
            resume=resume,
        )
        pbar.update()


def train(
    SIE_id,
    training_dir,
    dataset_file,
    imgsz=512,
    epochs=300,
    batch=256,
    iou=0.5,
    stop_training=False,
    devices=None,
    exist_ok=False,
    mode_text=True,
    resume=False,
):
    # model = YOLO('yolov8n.yaml')
    name = "train_wo{}".format(SIE_id)
    if resume:
        model = YOLO(os.path.join(training_dir, name, "weights", "last.pt"))
    else:
        model = YOLO("yolov8l.yaml")

    patience = 1000 if not stop_training else 75

    results = model.train(
        pretrained=False,
        val=True,
        data=dataset_file,
        project=training_dir,
        name=name,
        epochs=epochs,
        imgsz=imgsz,
        deterministic=True,
        iou=iou,
        device=devices,
        batch=batch,
        patience=patience,
        plots=True,
        visualize=True,
        workers=8,
        resume=resume,
        exist_ok=exist_ok,
    )  # resume)
    try:
        results_path = os.path.join(training_dir, name, "results.csv")
        plot_results(results_path, segment=True)
    except:
        print("No path found to store results.csv")

    return model
