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


import os
import sys
import json
import yaml
import shutil
import glob
import copy
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy
import torch
import torchvision
import torchmetrics
import cv2
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.plotting import plot_results
from ultralytics.utils.plotting import plt_color_scatter
from collections import OrderedDict

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

from ..modules import output

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


def get_predictions(model, image_list):
    results = model.track(image_list, verbose=False)
    return results


def calculate_pr(tp, fp, fn):
    try:
        precision = tp / (tp + fp)
    except:
        precision = 0
    try:
        recall = tp / (tp + fn)
    except:
        recall = 0
    return precision, recall


def calculate_iou(label_dict, pred_dict, label_id, pred_id):
    iou_x1 = np.max([label_dict[label_id]["x1"], pred_dict[pred_id]["x1"]])
    iou_y1 = np.max([label_dict[label_id]["y1"], pred_dict[pred_id]["y1"]])
    iou_x2 = np.min([label_dict[label_id]["x2"], pred_dict[pred_id]["x2"]])
    iou_y2 = np.min([label_dict[label_id]["y2"], pred_dict[pred_id]["y2"]])

    label_area = (label_dict[label_id]["x2"] - label_dict[label_id]["x1"]) * (
        label_dict[label_id]["y1"] - label_dict[label_id]["y2"]
    )
    pred_area = (pred_dict[pred_id]["x2"] - pred_dict[pred_id]["x1"]) * (
        pred_dict[pred_id]["y1"] - pred_dict[pred_id]["y2"]
    )
    intersection_area = (iou_x2 - iou_x1) * (iou_y1 - iou_y2)
    union_area = label_area + pred_area - intersection_area

    if (iou_y1 > iou_y2) or (iou_x1 > iou_x2):
        return None
    return intersection_area / union_area


def count_tpfpfn(label_matches, pred_matches):
    tp_sample = 0
    fp_sample = 0
    fn_sample = 0

    for label_id in label_matches:
        outcome = label_matches[label_id]["outcome"]
        if outcome == "tp":
            tp_sample += 1
        elif outcome == "fn" or outcome == "fn_misclass":
            fn_sample += 1
    for pred_id in pred_matches:
        outcome = pred_matches[pred_id]["outcome"]
        if outcome == "fp":
            fp_sample += 1

    return tp_sample, fp_sample, fn_sample


def standardize_classifications(result, labels, tile_id):
    pred_dict = {"pred_" + str(i): {} for i in range(len(result))}
    label_dict = {"label_" + str(i): {} for i in range(len(labels))}
    for i in range(len(result)):
        result_dict = result[i].tojson().split("[")[1].split("]")[0]
        result_dict = json.loads(result_dict)
        key_id = "pred_" + str(i)
        pred_dict[key_id]["predicted class"] = result_dict["class"]
        pred_dict[key_id]["x1"] = result_dict["box"]["x1"]
        pred_dict[key_id]["y1"] = result_dict["box"]["y1"]
        pred_dict[key_id]["x2"] = result_dict["box"]["x2"]
        pred_dict[key_id]["y2"] = result_dict["box"]["y2"]
        pred_dict[key_id]["tile_id"] = tile_id
    for i, obj in enumerate(labels):
        xcn = float(obj[1])
        ycn = float(obj[2])
        wn = float(obj[3])
        hn = float(obj[4])
        key_id = "label_" + str(i)
        label_dict[key_id]["class"] = int(obj[0])
        label_dict[key_id]["x1"] = 512 * (xcn - (wn / 2))
        label_dict[key_id]["y1"] = 512 * (ycn - (hn / 2))
        label_dict[key_id]["x2"] = 512 * (xcn + (wn / 2))
        label_dict[key_id]["y2"] = 512 * (ycn + (hn / 2))
        label_dict[key_id]["tile_id"] = tile_id

    return label_dict, pred_dict


def label_match(
    label_dict,
    pred_dict,
    result=None,
    iou_threshold=0.75,
    model_performance=None,
    verbose=False,
):
    label_matches = {label_id: {"outcome": "fn"} for label_id in label_dict}
    pred_matches = {pred_id: {"outcome": "fp"} for pred_id in pred_dict}

    for label_id in label_dict:
        for pred_id in pred_dict:
            iou_area = calculate_iou(label_dict, pred_dict, label_id, pred_id)
            # plot_intersection(result, i, iou_x1, iou_y1, iou_x2, iou_y2)

            if iou_area is None:
                pass
            elif (iou_area > iou_threshold) and (
                label_dict[label_id]["class"] == pred_dict[pred_id]["predicted class"]
            ):
                # TRUE POSITIVE CASE
                if (label_matches[label_id]["outcome"] != "tp") and (
                    pred_matches[pred_id]["outcome"] != "tp"
                ):
                    label_matches[label_id] = {
                        "outcome": "tp",
                        "iou": iou_area,
                        "associated_prediction": pred_id,
                    }
                    pred_matches[pred_id] = {
                        "outcome": "tp",
                        "iou": iou_area,
                        "associated_label": label_id,
                    }
                elif (label_matches[label_id]["outcome"] == "tp") and (
                    pred_matches[pred_id]["outcome"] == "tp"
                ):
                    if iou_area > label_matches[label_id]["iou"]:
                        print("OVERWRITE")
                        pred_id_temp = label_matches[label_id]["associated_prediction"]
                        label_matches[label_id] = {
                            "outcome": "tp",
                            "iou": iou_area,
                            "associated_prediction": pred_id,
                        }
                        pred_matches[pred_id] = {
                            "outcome": "tp",
                            "iou": iou_area,
                            "associated_label": label_id,
                        }
                        pred_matches[pred_id_temp] = {
                            "outcome": "fp",
                            "iou": iou_area,
                            "associated_label": label_id,
                        }
                # assert label_id == pred_matches[pred_id]['associated_label']
            else:
                # FALSE POSITIVE/MISCLASSIFICATION
                if (
                    label_matches[label_id]["outcome"] != "tp"
                    and pred_matches[pred_id]["outcome"] != "tp"
                ):
                    label_matches[label_id] = {
                        "outcome": "fn_misclass",
                        "iou": iou_area,
                        "associated_prediction": pred_id,
                    }

    return label_matches, pred_matches


def score_json_overall(tp_all, fp_all, fn_all, pred_num, gt_num):
    precision, recall = calculate_pr(tp_all, fp_all, fn_all)
    return {
        "tp": tp_all,
        "fp": fp_all,
        "fn": fn_all,
        "predictions": pred_num,
        "ground truth": gt_num,
        "precision": precision,
        "recall": recall,
    }


def score_json_single_sample(tp, fp, fn):
    precision, recall = calculate_pr(tp, fp, fn)
    return {"P": precision, "R": recall}


# Workhorse function


def score_preds(
    results, interaction_id, data_dir, in_ex, iou_threshold=0.7, model_performance=None
):
    tp_all_sample = 0  # All label->predictions
    fp_all_sample = 0
    fn_all_sample = 0
    pred_num = 0
    gt_num = 0

    # model_performance={}
    # For result in results:
    for predictions in results:
        image_path = predictions.path
        tile_id = os.path.basename(image_path).split(".")[0]
        label_path = os.path.join(data_dir, "labels", "{}.txt".format(tile_id))

        with open(label_path) as f:
            labels = [obj.split("\n")[0].split(" ") for obj in list(f)]

        label_dict, pred_dict = standardize_classifications(
            predictions, labels, tile_id
        )
        label_matches, pred_matches = label_match(
            label_dict,
            pred_dict,
            predictions,
            iou_threshold=iou_threshold,
            verbose=False,
        )
        tp_sample, fp_sample, fn_sample = count_tpfpfn(label_matches, pred_matches)

        per_sample_perf = score_json_single_sample(tp_sample, fp_sample, fn_sample)
        model_performance[in_ex]["Per Sample Performance"][tile_id] = per_sample_perf

        tp_all_sample += tp_sample
        fp_all_sample += fp_sample
        fn_all_sample += fn_sample
        pred_num += len(pred_dict)
        gt_num += len(label_dict)

        # print(tile_id)
        # print(pred_matches)
        # print(label_matches)

    perf_overall = score_json_overall(
        tp_all_sample, fp_all_sample, fn_all_sample, pred_num, gt_num
    )
    model_performance[in_ex]["Overall Performance"] = perf_overall

    return perf_overall, model_performance


def plot_intersection(result, i, iou_x1, iou_y1, iou_x2, iou_y2):
    tile_id = os.path.basename(result.path).split(".")[0]
    img = cv2.imread(result.path)
    boxes = result.boxes.cpu().numpy()
    for k, box in enumerate(boxes):
        r = box.xyxy[0].astype(int)
        crop = img[int(iou_y1) : int(iou_y2), int(iou_x1) : int(iou_x2)]
        print(crop)
        cv2.imwrite(
            "./img/" + tile_id + "_img_" + str(i) + "_pred" + str(k) + ".jpg", crop
        )


def write_performance_file_single(performance_json, SIE_id, SIE_performance_dir):
    model_performance_output = json.dumps(
        performance_json, sort_keys=False, indent=4, separators=(",", ": ")
    )
    with open(
        os.path.join(SIE_performance_dir, "performance{}.json".format(SIE_id)), "w"
    ) as f:
        f.write(model_performance_output)

    return model_performance_output


def write_performance_file_aggregate_table(
    performance_json, config_dir, SIE_performance_dir, whole=True
):
    with open(os.path.join(config_dir, "coverage.json")) as f:
        universe = json.load(f)["universe"]

    if whole:
        columns = [
            "Model",
            "Feature_Excluded",
            "Level_Excluded",
            "Included_in_Training",
            "Precision",
            "Recall",
            "F1",
            "Test_Set_Size",
        ]
    else:
        columns = [
            "Model",
            "Feature Excluded",
            "Level Excluded",
            "Included in Training",
            "Precision",
            "Recall",
            "F1",
            "Test Set Size",
        ]
    SIE_table = pd.DataFrame()  # , index=index)

    for k, SIE_id in enumerate(performance_json):
        SIE_id_split = SIE_id.split("_")
        feature_excl = SIE_id_split[1]
        level_excl = SIE_id_split[2]
        if "x" in SIE_id:
            feature_name = "BASELINE"
            level_name = None
        else:
            feature_name = universe["features"][int(feature_excl)]
            level_name = universe["levels"][int(feature_excl)][int(level_excl)]

        for inex in performance_json[SIE_id]:
            if "x" in SIE_id and "excluded" in inex:
                continue
            p = performance_json[SIE_id][inex]["precision"]
            r = performance_json[SIE_id][inex]["recall"]
            n = performance_json[SIE_id][inex]["test_set_size"]
            f1 = (2 * p * r) / (p + r)
            partition = "Yes" if ("included" in inex) else "No"

            values = [
                "Model {}".format(k),
                feature_name,
                level_name,
                partition,
                p,
                r,
                f1,
                n,
            ]
            row = {k: v for k, v in zip(columns, values)}
            SIE_table = SIE_table.append(row, ignore_index=True)

    SIE_table.to_csv(os.path.join(SIE_performance_dir, "aggregate_performance.csv"))
    output.output(
        performance_json,
        os.path.join(SIE_performance_dir, "aggregate_performance.json"),
    )

    return performance_json


def add_entry_aggregate(
    model_performance_aggregate,
    interaction_id,
    performance_incl,
    performance_excl,
    incl_size,
    excl_size,
):
    performance_incl["test_set_size"] = incl_size
    performance_excl["test_set_size"] = excl_size

    model_performance_aggregate[interaction_id] = {
        "test_included": None,
        "test_excluded": None,
    }
    model_performance_aggregate[interaction_id]["test_included"] = performance_incl
    model_performance_aggregate[interaction_id]["test_excluded"] = performance_excl

    return model_performance_aggregate


def evaluate(
    SIE_splits,
    data_dir,
    dataset_dir,
    training_dir,
    config_dir,
    iou_threshold=0.7,
    table=True,
    verbose=True
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