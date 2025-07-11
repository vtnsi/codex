import glob
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.plotting import plt_color_scatter
from ultralytics.utils.plotting import plot_results

import typing_extensions
import importlib

importlib.reload(typing_extensions)


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


def evaluate(
    SIE_splits,
    data_dir,
    dataset_dir,
    training_dir,
    config_dir,
    iou_threshold=0.7,
    table=True,
    verbose=True,
):
    model_paths = glob.glob(os.path.join(training_dir, "*train*", "weights", "best.pt"))
    pbar = tqdm(model_paths, leave=False)
    model_performance_aggregate = {}
    print(
        "Out of {} designated splits, found {} trained models in {}.".format(
            len(SIE_splits), len(model_paths), training_dir
        )
    )

    SIE_performance_dir = os.path.join(config_dir, "SIE_performance")
    if not os.path.exists(SIE_performance_dir):
        os.makedirs(SIE_performance_dir)

    for i, SIE_id in enumerate(SIE_splits):
        path = os.path.abspath(
            os.path.join(
                training_dir, "train_wo{}".format(SIE_id), "weights", "best.pt"
            )
        )
        try:
            model = YOLO(path)
        except:
            print("Model {} doesn't exist. Consider retraining".format(SIE_id))
            continue

        test_in_img_paths, test_ex_img_paths = get_image_paths(SIE_id, dataset_dir)
        if verbose:
            print(len(test_in_img_paths), len(test_ex_img_paths))

        model_performance_single = {
            "split_id": SIE_id,
            "test_included": {"Overall Performance": {}, "Per Sample Performance": {}},
            "test_excluded": {"Overall Performance": {}, "Per Sample Performance": {}},
        }
        results_covered = get_predictions(model, test_in_img_paths)

        # KEY SCORING FUNCTIONS
        performance_covered, model_performance_single = score_preds(
            results_covered,
            SIE_id,
            data_dir,
            "test_included",
            model_performance=model_performance_single,
            iou_threshold=iou_threshold,
        )
        performance_withheld = {}  # Should be overwritten verwritten if not baseline model

        if SIE_id != "_x_":
            # TEST SET WITH ONLY INTERACTION
            results_withheld = get_predictions(model, test_ex_img_paths)
            performance_withheld, model_performance_single = score_preds(
                results_withheld,
                SIE_id,
                data_dir,
                "test_excluded",
                model_performance=model_performance_single,
                iou_threshold=iou_threshold,
            )

        write_performance_file_single(
            model_performance_single, SIE_id, SIE_performance_dir
        )
        add_entry_aggregate(
            model_performance_aggregate,
            SIE_id,
            performance_covered,
            performance_withheld,
            len(test_in_img_paths),
            len(test_ex_img_paths),
        )
        pbar.update()

    if table:
        write_performance_file_aggregate_table(
            model_performance_aggregate, config_dir, SIE_performance_dir
        )

    return model_performance_aggregate
