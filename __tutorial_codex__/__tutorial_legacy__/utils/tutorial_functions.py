from PIL import Image
import json
import numpy as np
import pandas as pd
import src.output as output
import CODEX as codex
import sys
import os

sys.path.append("../../")


def __assembly_info_user__(path=None):
    if path is None:
        path = "tutorial_materials"
    print(os.getcwd())
    print(
        "TUTORIAL MODE MATERIALS. SELECT ITEMS FROM HERE OR PROVIDE YOUR OWN ITEMS CORRECTLY: ",
        os.listdir(path),
    )


def dsc_all(input_comparison):
    SIE_perf = glob.glob(
        os.path.join("tutorial_materials", "performance", "performance_*_*.json")
    )
    print(SIE_perf)
    split_ids = []

    for path in SIE_perf:
        print(path)
        if os.path.basename(path) == "split_00.json":
            continue
        with open(path) as f:
            split_ids.append(json.load(f)["split_id"])

    print(split_ids)
    # split_ids.remove('split_00.json')
    split_ids = pd.unique(split_ids)
    split_ids.sort()

    print(split_ids)
    print(len(split_ids))

    orig_id = input_comparison["config_id"]
    for id in split_ids:
        print(id)
        """if os.path.exists(os.path.join(input_comparison['codex_directory'], input_comparison['config_id'])):
            print('continuing')
            continue"""

        input_comparison["config_id"] = orig_id + id

        input_comparison["split_file"] = [
            "model{}incl.json".format(id),
            "model{}excl.json".format(id),
            "model{}all.json".format(id),
            "model_x_all.json",
        ]
        input_comparison["performance_file"] = [
            "performance{}incl.json".format(id),
            "performance{}excl.json".format(id),
            "performance{}all.json".format(id),
            "performance_x_all.json",
        ]

        codex.run(input_comparison)


def pathway_check_recursive(path, key, prefix=""):
    path = os.path.join(prefix, path)

    if not os.path.exists(path):
        new_path = input("Path not found. Specify location for {}".format(key))
        pathway_check_recursive(new_path, key, prefix)
    else:
        return path


def assemble_input_file_user(codex_input_demo):
    __assembly_info_user__()

    modes = [
        "dataset evaluation",
        "dataset split evaluation",
        "dataset split comparison",
        "performance by interaction",
        "model probing",
    ]
    mode = input("CODEX MODE: ")
    if mode not in modes:
        mode = input("CODEX mode not found. Supported modes: {}".format(modes))

    # MODE GENERIC PARAMS
    codex_input_demo["mode"] = mode
    codex_dir = input(
        "CODEX DIRECTORY: (default: {})".format(codex_input_demo["codex_directory"])
    )
    if codex_dir != "":
        codex_input_demo["codex_directory"] = codex_dir
    else:
        codex_dir = "./"
    print("Selected {} as CODEX directory.".format(codex_dir))
    codex_input_demo["config_id"] = input(
        "CONFIG ID (currently set to '{}')".format(codex_input_demo["config_id"])
    )
    print("Selected {} as experiment config ID.".format(codex_input_demo["config_id"]))

    data_dir = input("DATASET DIRECTORY (within codex directory): ")
    data_dir = pathway_check_recursive(data_dir, "dataset_directory")
    dataset_filename = input("DATASET FILE NAME:")
    dataset_fp = os.path.join(data_dir, dataset_filename)
    dataset_filename = pathway_check_recursive(
        dataset_filename, "dataset_file", prefix=data_dir
    )

    codex_input_demo["data_directory"] = data_dir
    codex_input_demo["dataset_file"] = dataset_filename

    if os.path.exists(os.path.join(data_dir, dataset_filename)):
        codex_input_demo["dataset_file"] = dataset_filename
    else:
        raise FileNotFoundError("Dataset file not found!")

    codex_input_demo["sample_id_column"] = input("SAMPLE ID COLUMN NAME: ")
    codex_input_demo["label_column"] = input("LABEL COLUMN NAME: ")
    codex_input_demo["features"] = (
        input("FEATURES (separate by commas, ','):").strip(" ").split(",")
    )

    bin_path = input("BINNING FILE PATHWAY (within codex directory): ")
    if bin_path == "":
        codex_input_demo["bin_filename"] = None
    else:
        codex_input_demo["bin_filename"] = pathway_check_recursive(
            bin_path, "bin_filename"
        )

    universe_path = input(
        "UNIVERSE FILE PATHWAY (within codex directory) (enter nothing for no predefined universe): "
    )
    if universe_path == "":
        codex_input_demo["universe"] = None
    else:
        codex_input_demo["universe"] = pathway_check_recursive(
            universe_path, "universe file"
        )

    codex_input_demo["use_augmented_universe"] = (
        input("AUGMENT UNIVERSE (y/n): ") == "y"
    )
    codex_input_demo["save_universe"] = input("SAVE UNIVERSE (y/n): ") == "y"
    codex_input_demo["timed_output"] = input("TIMESTAMP OUTPUT FOLDER (y/n): ") == "y"

    if mode == "dataset split evaluation" or mode == "performance by interaction":
        pass
    elif mode == "dataset split comparison" or mode == "model_probing":
        pass

    return codex_input_demo


def gen_splits():
    # SPLIT GEN DELETE AFTER USE
    num_splits = 3
    for i in range(num_splits):
        dataset_full = dataset_full.sample(len(dataset_full))
        id_list = dataset_full.id.tolist()
        n = len(id_list)
        train_list = [str(i) for i in id_list[: int(0.65 * n)]]
        val_list = [str(i) for i in id_list[int(0.65 * n) : int(0.80 * n)]]
        test_list = [str(i) for i in id_list[int(0.8 * n) :]]

        assert n == (len(train_list + val_list + test_list))

        split_dict = {"train": train_list, "validation": val_list, "test": test_list}
        performance_dict = {
            "split_file": "split_0{}.json".format(i),
            "test": {"Overall Performance": {"accuracy": random.uniform(0, 1)}},
        }


def probe_splits():
    # DELETE AFTER USE
    with open("tutorial_materials/splits/split_00.json") as f:
        split_00 = json.load(f)
        test_ids = split_00["test"]
        probe_ids = test_ids[: int(len(test_ids) / 2)]
        exploit_ids = test_ids[int(len(test_ids) / 2) : len(test_ids)]

    split_99 = {
        "train": split_00["train"],
        "validation": split_00["validation"],
        "test_probe": probe_ids,
        "test_exploit": exploit_ids,
    }
    output.output_json_readable(
        split_99,
        write_json=True,
        file_path=os.path.join("tutorial_materials/splits/split_99.json"),
    )

    test_ids.sort()

    performance_99 = performance_dict = {
        "split_id": "split_99.json",
        "test_probe": {
            "Overall Performance": {"accuracy": None},
            "Per Sample Performance": {},
        },
        "test_exploit": {
            "Overall Performance": {"accuracy": None},
            "Per Sample Performance": {},
        },
    }

    values = [0.0, 1.0, 1.0, 1.0]
    for i, id in enumerate(test_ids):
        if id in probe_ids:
            performance_99["test_probe"]["Per Sample Performance"][id] = {
                "accuracy": random.sample(values, 1)[0]
            }
        else:
            assert id in exploit_ids
            performance_99["test_exploit"]["Per Sample Performance"][id] = {
                "accuracy": random.sample(values, 1)[0]
            }

    performance_99["test_probe"]["Overall Performance"]["accuracy"] = np.mean(
        [
            performance_99["test_probe"]["Per Sample Performance"][id]["accuracy"]
            for id in performance_99["test_probe"]["Per Sample Performance"]
        ]
    )
    performance_99["test_exploit"]["Overall Performance"]["accuracy"] = np.mean(
        [
            performance_99["test_exploit"]["Per Sample Performance"][id]["accuracy"]
            for id in performance_99["test_exploit"]["Per Sample Performance"]
        ]
    )

    output.output_json_readable(
        performance_99,
        write_json=True,
        file_path="./tutorial_materials/performance/performance_99-ps.json",
    )


def psperf():
    with open("./tutorial_materials/splits/split_00.json") as f:
        split_00 = json.load(f)
    test_ids = split_00["test"]

    test_ids.sort()

    performance_dict = {
        "split_id": "split_00.json",
        "test": {
            "Overall Performance": {"accuracy": None},
            "Per Sample Performance": {},
        },
    }

    print(test_ids)
    values = [0.0, 1.0, 1.0, 1.0]
    for i, id in enumerate(test_ids):
        performance_dict["test"]["Per Sample Performance"][id] = {
            "accuracy": random.sample(values, 1)[0]
        }


def display_experiment_setup(codex_input):
    return


def display_selected_result(results_dict, element, strengths, **kwargs):
    for t in strengths:
        print(results_dict[t])

    return


def display_results(results_dict, element, strengths, **kwargs):
    # element: ranks, missing interactions
    if element == "SDCC":
        sdcc_element = kwargs["sdcc_element"]
        assert sdcc_element in [
            "train",
            "validation",
            "test",
            "train-test",
            "test-train",
            "validation-test",
            "test-validation",
        ]
        # call
    else:
        pass
