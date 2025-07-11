import os
import pandas as pd
import json
import numpy as np
import random
import glob

df = pd.read_csv("datasets/pdb_data_no_dups-no_na.csv")
"""
print(df['macromoleculeType'].value_counts())
print(df['crystallizationMethod'].value_counts())"""


print(df["phValue"].describe())
print(df["residueCount"].describe())


exit()


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


with open("splits/split_rareplanes_example.json") as s:
    split = json.load(s)


test_ids = split["test"]

perf = {
    "Overall Performance": {},
    "Per Sample Performance": {
        test_id: {
            "precision": round(random.random(), 3),
            "recall": round(random.random(), 3),
        }
        for test_id in test_ids
    },
}

perf["Overall Performance"]["precision"] = np.mean(
    [
        perf["Per Sample Performance"][test_id]["precision"]
        for test_id in perf["Per Sample Performance"]
    ]
)
perf["Overall Performance"]["recall"] = np.mean(
    [
        perf["Per Sample Performance"][test_id]["recall"]
        for test_id in perf["Per Sample Performance"]
    ]
)

output_json_readable(
    perf, write_json=True, file_path="performance/performance_rareplanes_example.json"
)
