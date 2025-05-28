import src.output as output
import os
import json
from glob import glob
import sys

sys.path.append("../../..")

SIE_splits = glob(
    os.path.join(
        "../../../../codex-use_cases/rareplanesSIE/exp-SIE_E2E/splits_by_json/model*.json"
    )
)
# new_dir = output.make_output_dir_nonexist(os.path.join('../../../../codex-use_cases/rareplanesSIE/splits-partitioned_e2e/'))
SIE_perf = glob(
    os.path.join(
        "../../../../codex-use_cases/rareplanesSIE/exp-SIE_E2E/SIE_performance/performance*.json"
    )
)

new_dir_split = output.make_output_dir_nonexist(
    os.path.join("../../../../codex-use_cases/rareplanesSIE/splits-consolidated/")
)  # os.path.join('../splits')
new_dir_perf = output.make_output_dir_nonexist(
    os.path.join("../../../../codex-use_cases/rareplanesSIE/performance-consolidated/")
)  # os.path.join('../performance')


def partition():
    for path in SIE_splits:
        with open(path) as f:
            split = json.load(f)

        split_incl = {
            "train": split["train"],
            "validation": split["validation"],
            "test": split["test_included"],
        }

        split_excl = {
            "train": split["train"],
            "validation": split["validation"],
            "test": split["test_excluded"],
        }

        filename = os.path.basename(path).split(".")[0]

        output.output_json_readable(
            split_incl,
            write_json=True,
            file_path=os.path.join(new_dir_split, "{}incl.json".format(filename)),
        )
        output.output_json_readable(
            split_excl,
            write_json=True,
            file_path=os.path.join(new_dir_split, "{}excl.json".format(filename)),
        )


def partition_perf():
    for path in SIE_perf:
        print(path)
        with open(path) as f:
            perf = json.load(f)

        perf_incl = {"split_id": perf["split_id"], "test": perf["test_included"]}

        perf_excl = {"split_id": perf["split_id"], "test": perf["test_excluded"]}

        filename = os.path.basename(path).split(".")[0]

        output.output_json_readable(
            perf_incl,
            write_json=True,
            file_path=os.path.join(new_dir_perf, "{}incl.json".format(filename)),
        )
        output.output_json_readable(
            perf_excl,
            write_json=True,
            file_path=os.path.join(new_dir_perf, "{}excl.json".format(filename)),
        )


def consolidate():
    for path in SIE_splits:
        with open(path) as f:
            split = json.load(f)

        new_split = {
            "train": split["train"],
            "validation": split["validation"],
            "test": split["test_excluded"] + split["test_included"],
        }

        filename = os.path.basename(path).split(".")[0]

        output.output_json_readable(
            new_split,
            write_json=True,
            file_path=os.path.join(new_dir_split, "{}all.json".format(filename)),
        )


def consolidate_perf():
    for path in SIE_perf:
        with open(path) as f:
            perf = json.load(f)

        if "_x_" in path:
            new_perf = {
                "split_id": perf["split_id"],
                "test": {
                    "Overall Performance": perf["test_included"]["Overall Performance"],
                    "Per Sample Performance": perf["test_included"][
                        "Per Sample Performance"
                    ],
                },
            }
        else:
            perf["test_included"]["Per Sample Performance"].update(
                perf["test_excluded"]["Per Sample Performance"]
            )

            tp = (
                perf["test_included"]["Overall Performance"]["tp"]
                + perf["test_excluded"]["Overall Performance"]["tp"]
            )
            fp = (
                perf["test_included"]["Overall Performance"]["fp"]
                + perf["test_excluded"]["Overall Performance"]["fp"]
            )
            fn = (
                perf["test_included"]["Overall Performance"]["fn"]
                + perf["test_excluded"]["Overall Performance"]["fn"]
            )
            pred = (
                perf["test_included"]["Overall Performance"]["predictions"]
                + perf["test_excluded"]["Overall Performance"]["predictions"]
            )
            gt = (
                perf["test_included"]["Overall Performance"]["ground truth"]
                + perf["test_excluded"]["Overall Performance"]["ground truth"]
            )

            prec = tp / (tp + fp)
            rec = tp / (tp + fn)

            new_perf = {
                "split_id": perf["split_id"],
                "test": {
                    "Overall Performance": {
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "predictions": pred,
                        "ground truth": gt,
                        "precision": prec,
                        "recall": rec,
                    },
                    "Per Sample Performance": perf["test_included"][
                        "Per Sample Performance"
                    ],
                },
            }

        filename = os.path.basename(path).split(".")[0]

        output.output_json_readable(
            new_perf,
            write_json=True,
            file_path=os.path.join(new_dir_perf, "{}all.json".format(filename)),
        )


# partition()
# partition_perf()
consolidate()
consolidate_perf()
