# Author Brian Lee leebri2n@vt.edu
# Created Date: Mar 1 2023
# Updated Date: 3 April, 2024
# Movement to new repo: 12 Apr, 2024
# Refactoring: 06 June, 2025

import os
import json
import matplotlib.pyplot as plt
import math
from datetime import datetime
import logging
import glob

from inspect import currentframe, getframeinfo
from vis import plotting
from output import results

LOGGER_OUT = logging.getLogger(__name__)

showfig = False
savefig = True
timed = False


def logger_parameters(verbosity: str, output_dir="", timed=True):
    logging.addLevelName(15, "CODEX_DEBUG")
    logging.addLevelName(25, "CODEX_INFO")

    if timed:
        timestamp = datetime.now().strftime("%Y_%m_%d-%I:%M")
    else:
        timestamp = ""

    if verbosity == "2":
        level = logging.getLevelName(15)
        levelnum = 15
    elif verbosity == "1":
        level = logging.getLevelName(25)
        levelnum = 25
    else:
        raise ValueError(
            "No logging level {} found for CODEX. Levels '1' or '2' supported.".format(
                verbosity
            )
        )

    os.makedirs(os.path.abspath(output_dir), exist_ok=True)
    if timed:
        filename = os.path.abspath(
            os.path.join(
                output_dir,
                "codex_out_{}-{}.log".format(level.split("_")[-1], timestamp),
            )
        )
    else:
        filename = os.path.abspath(
            os.path.join(output_dir, "codex_out_{}.log".format(level.split("_")[-1]))
        )

    return levelnum, filename


def intialize_logger(logger_name, levelnum, filename):
    logging.basicConfig(filename=filename, level=levelnum)
    logger_for_file = logging.getLogger(logger_name)
    logger_for_file.setLevel(levelnum)
    logger_for_file.log(level=levelnum, msg=f"Logger for {logger_name} intialized.")

    return levelnum


# NOTE: DEPRECATE subset variable in pbi, dataset eval


def output_json_readable(
    json_obj: dict | str,
    print_json=False,
    write_json=False,
    file_path="",
    sort=False,
    truncate_lists=False,
):
    """
    Formats JSON object to human-readable format with print/save options.
    """
    if type(json_obj) is str:
        with open(file_path) as f:
            json_obj = json.load(json_str)

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


def dataset_eval_vis(output_dir, coverage_results):
    cc_output_dir = create_output_dir(os.path.join(output_dir, "CC"))

    strengths = coverage_results["info"]["t"]
    for t in strengths:
        LOGGER_OUT.log(
            level=25,
            msg="CC: t = {}\n{}".format(
                t, [coverage_results["results"][t]["CC"] for t in strengths]
            ),
        )

        plotting.coverage_map("binary", coverage_results, t, cc_output_dir)
        plotting.coverage_map("frequency", coverage_results, t, cc_output_dir)
        plotting.coverage_map(
            "proportion_frequency", coverage_results, t, cc_output_dir
        )
        plotting.coverage_map(
            "proportion_frequency_standardized", coverage_results, t, cc_output_dir
        )

    print("reached")
    return output_json_readable(
        coverage_results,
        write_json=True,
        file_path=os.path.join(output_dir, "coverage.json"),
    )


def dataset_split_eval_vis(
    output_dir, coverage_results, split_id=None, comparison=None
):
    if comparison:
        output_dir = create_output_dir(os.path.join(output_dir, split_id))
    cc_output_dir = create_output_dir(os.path.join(output_dir, "CC"))
    sdcc_output_dir = create_output_dir(os.path.join(output_dir, "SDCC"))

    strengths = coverage_results["info"]["t"]
    for t in strengths:
        for coverage_subset in coverage_results["results_all"]:
            coverage_results_subset = {
                "info": coverage_results["info"],
                "universe": coverage_results["universe"],
                "results": coverage_results["results_all"][coverage_subset]["results"],
            }
            print(
                getframeinfo(currentframe()).filename,
                getframeinfo(currentframe()).lineno,
            )
            output_json_readable(coverage_results_subset, print_json=True)

            if "-" in coverage_subset:
                # SDCC
                plotting.coverage_map(
                    "sdcc_binary_constraints",
                    coverage_results_subset,
                    t,
                    sdcc_output_dir,
                    direction=coverage_subset,
                    coverage_subset=coverage_subset,
                )
            else:
                plotting.coverage_map(
                    "binary",
                    coverage_results_subset,
                    t,
                    cc_output_dir,
                    coverage_subset=coverage_subset,
                )
                plotting.coverage_map(
                    "frequency",
                    coverage_results_subset,
                    t,
                    cc_output_dir,
                    coverage_subset=coverage_subset,
                )
                plotting.coverage_map(
                    "proportion_frequency",
                    coverage_results_subset,
                    t,
                    cc_output_dir,
                    coverage_subset=coverage_subset,
                )
                plotting.coverage_map(
                    "proportion_frequency_standardized",
                    coverage_results_subset,
                    t,
                    cc_output_dir,
                    coverage_subset=coverage_subset,
                )
            plt.close("all")

        """plotting.split_elements_bar(
            output_dir,
            coverage_results["info"]["dataset_name"],
            coverage_results,
            t,
            split_id,
            sdcc_directions,
        )"""
    plt.close("all")

    print(f"results saved to {output_dir}")
    return output_json_readable(
        coverage_results,
        write_json=True,
        file_path=os.path.join(output_dir, "coverage.json"),
    )


def performance_by_interaction_vis(
    output_dir,
    coverage_results,
    display_interaction_order="ascending",
    display_interaction_num=10,
    coverage_subset="train",
):
    cc_output_dir = create_output_dir(os.path.join(output_dir, "CC"))

    strengths = coverage_results["info"]["t"]
    metrics = coverage_results["info"]["metrics"]

    for t in strengths:
        for metric in metrics:
            plotting.coverage_map(
                "binary",
                coverage_results,
                t,
                output_dir=cc_output_dir,
                coverage_subset=coverage_subset,
            )
            plotting.coverage_map(
                "frequency",
                coverage_results,
                t,
                output_dir=cc_output_dir,
                coverage_subset=coverage_subset,
            )
            plotting.coverage_map(
                "proportion_frequency",
                coverage_results,
                t,
                output_dir=cc_output_dir,
                coverage_subset=coverage_subset,
            )
            plotting.coverage_map(
                "proportion_frequency_standardized",
                coverage_results,
                t,
                output_dir=cc_output_dir,
                coverage_subset=coverage_subset,
            )

            # print(coverage_results['results']['performance'])
            plotting.coverage_map(
                "performance",
                coverage_results,
                t,
                metric=metric,
                output_dir=cc_output_dir,
                coverage_subset=coverage_subset,
                pi_perf=coverage_results["results"]["performance"],
            )

            """interactions_consolidated = results.consolidated_interaction_info(
                coverage_results, strengths, metric, order=order, display_n=display_n
            )
            plotting.plot_pbi_bar(
                output_dir, interactions_consolidated, t, metric, display_n, order
            )

            human_readable = coverage_results['results'][t]["human readable performance"][metric]
            plotting.plot_pbi_frequency_scatter(
                "all",
                output_dir,
                dataset_name,
                counts,
                perf,
                human_readable,
                t,
                metric,
                ranks,
                display_all_lr,
            )
            plotting.plot_pbi_frequency_scatter(
                "signif",
                output_dir,
                dataset_name,
                counts,
                perf,
                human_readable,
                t,
                metric,
                ranks,
                display_all_lr,
            )
            plotting.plot_pbi_frequency_scatter(
                "highlights",
                output_dir,
                dataset_name,
                counts,
                perf,
                human_readable,
                t,
                metric,
                ranks,
                display_all_lr,
            )"""

        coverage_results = output_json_readable(
            coverage_results,
            write_json=True,
            file_path=os.path.join(output_dir, "coverage.json"),
        )
        plt.close("all")
    return coverage_results


def dataset_split_comp_vis(
    output_dir,
    dataset_name,
    coverage_multi,
    performance_multi,
    strengths,
    metric,
    split_ids=None,
):
    output_dir_outer = create_output_dir(output_dir)

    for split_id in split_ids:
        # for split in coverage_multi:
        cc_output_dir = create_output_dir(
            os.path.join(output_dir_outer, split_id, "CC")
        )
        sdcc_output_dir = create_output_dir(
            os.path.join(output_dir_outer, split_id, "SDCC")
        )

        coverage = coverage_multi[split_id]
        for t in strengths:
            sdcc_directions = [
                key for key in coverage[t] if "-" in key
            ]  # and 'val' not in key]
            for direction in sdcc_directions:
                plotting.split_comp_scatter(
                    output_dir_outer,
                    dataset_name,
                    coverage_multi,
                    performance_multi,
                    t,
                    direction,
                    metric,
                    split_ids=split_ids,
                )

    result = output_json_readable(
        coverage_multi,
        write_json=True,
        file_path=os.path.join(output_dir, "coverage_aggregate.json"),
    )
    plt.close("all")
    return result


def model_probing_vis(
    output_dir,
    name_p,
    name_e,
    coverage_probe_results,
    coverage_exploit_results,
    strengths,
    metric,
    order,
    display_n,
    subset,
    funct=[math.log10],
):
    output_dir = create_output_dir(output_dir)
    for t in strengths:
        performance_by_interaction_vis(
            os.path.join(output_dir, "probe"),
            name_p,
            coverage_probe_results,
            strengths,
            metric,
            order,
            display_n,
            funct,
            subset,
        )
        performance_by_interaction_vis(
            os.path.join(output_dir, "exploit"),
            name_e,
            coverage_exploit_results,
            strengths,
            metric,
            order,
            display_n,
            funct,
            subset,
        )

        interactions_probe = results.consolidated_interaction_info(
            coverage_probe_results, strengths, metric, order, display_n
        )
        interactions_exploit = results.consolidated_interaction_info(
            coverage_exploit_results, strengths, metric, order, display_n
        )
        plotting.plot_probe_exploit_suite(
            output_dir, interactions_probe, interactions_exploit, t
        )

    result = output_json_readable(
        {name_p: coverage_probe_results, name_e: coverage_exploit_results},
        write_json=True,
        file_path=os.path.join(output_dir, "coverage_combined.json"),
    )

    plt.close("all")
    return result


# -------------------------- WRITING -------------------------- #
# writes the CC for a t to the t file


def writeCCtToFile(output_dir, name, t, CC):
    output_dir = create_output_dir(os.path.join(output_dir, "CC"))
    filename = os.path.join(output_dir, "t{}_{}.txt".format(t, name))

    with open(filename, "w") as tfile:
        tfile.write(
            "interactions appearing in {}: {}\n".format(
                name, CC["countAppearingInteractions"]
            )
        )  # name, strcccount, \n
        tfile.write(
            "total possible interactions: {}\n".format(CC["totalPossibleInteractions"])
        )
        cc = CC["countAppearingInteractions"] / CC["totalPossibleInteractions"]
        tfile.write("CC({}): {}\n".format(name, cc))

    LOGGER_OUT.log(
        level=15,
        msg="interactions appearing in {}: {}\n".format(
            name, CC["countAppearingInteractions"]
        ),
    )
    LOGGER_OUT.log(
        level=15,
        msg="interactions appearing in {}: {}\n".format(
            name, CC["countAppearingInteractions"]
        ),
    )
    LOGGER_OUT.log(
        level=15,
        msg="total possible interactions: {}\n".format(CC["totalPossibleInteractions"]),
    )
    # print(tfile.write("CC({}): {}\n".format(name, cc)))


# writes the SDCC for a t to the t file
def writeSDCCtToFile(output_dir, sourcename, targetname, t, SDCC):
    output_dir = create_output_dir(os.path.join(output_dir, "SDCC"))
    filename = os.path.join(
        output_dir, "t{}_{}-{}_SDCC.txt".format(t, sourcename, targetname)
    )

    with open(filename, "w") as tfile:
        tfile.write(
            "interactions in {}: {}\n".format(targetname, SDCC["interactionsInTarget"])
        )
        tfile.write(
            "interactions in {} not in {}: {}\n".format(
                targetname, sourcename, SDCC["setDifferenceInteractions"]
            )
        )
        try:
            sdcc = SDCC["setDifferenceInteractions"] / SDCC["interactionsInTarget"]
        except:
            sdcc = "0 Interactions in target"
        "SDCC({}-{}): {}\n".format(targetname, sourcename, sdcc)
        tfile.write("SDCC({}-{}): {}\n".format(targetname, sourcename, sdcc))

    LOGGER_OUT.log(
        level=15,
        msg="Interactions in {}: {}\n".format(targetname, SDCC["interactionsInTarget"]),
    )
    LOGGER_OUT.log(
        level=15,
        msg="Interactions in {} not in {}: {}\n".format(
            targetname, sourcename, SDCC["setDifferenceInteractions"]
        ),
    )
    LOGGER_OUT.log(
        level=15, msg="SDCC({}-{}): {}\n".format(targetname, sourcename, sdcc)
    )


# writes the missing interactions for a t to the t file


def writeMissingtoFile(output_dir, name, t, decodedMissing):
    output_dir = create_output_dir(os.path.join(output_dir, "CC"))
    filename = os.path.join(output_dir, "t{}_{}_missing.txt".format(t, name))

    with open(filename, mode="w") as tfile:
        tfile.write("Missing interactions for strength " + str(t) + ":\n")
        for interaction in decodedMissing:
            tfile.write(str(interaction) + "\n")
        tfile.close()


def writeSetDifferencetoFile(
    output_dir, sourcename, targetname, t, setDifferenceInteractions
):
    output_dir = create_output_dir(os.path.join(output_dir, "SDCC"))
    filename = os.path.join(
        output_dir, "t{}_{}-{}_setDifference.txt".format(t, targetname, sourcename)
    )

    with open(filename, mode="w") as tfile:
        tfile.write("set difference interactions for strength {}:\n".format(t))
        for interaction in setDifferenceInteractions:
            tfile.write("{}\n".format(str(interaction)))


# first file write preserves pipeline, second file write used in flow for targeted retraining


def writeImagestoFile(output_dir, sourcename, targetname, t, setDifferenceImages):
    filename = (
        output_dir
        + str(t)
        + "_"
        + targetname
        + "_"
        + sourcename
        + "_setDifferenceImages.csv"
    )
    setDifferenceImages.to_csv(filename, index=False)
    filename = output_dir + targetname + "_SDImages.csv"
    setDifferenceImages.to_csv(filename, index=False)


def create_output_dir(output_dir_new, timed=False):
    """
    Makes a new output directory, either based on time or based on
    whether or not it already exists, using the existing directory if it does.
    """
    if timed:
        output_dir_new = "{}-{}".format(
            output_dir_new, datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        )

    if not os.path.exists(output_dir_new):
        os.makedirs(output_dir_new)

    return os.path.realpath(output_dir_new)


def SIE_regression_test_vis(
    output_dir, model_summary, contrasts_summary, contrast_names
):
    results = {
        "model": model_summary.as_text(),
        "contrasts": {"test": contrasts_summary.as_text(), "names": contrast_names},
    }
    results = output_json_readable(
        results, write_json=True, file_path=os.path.join(output_dir, "coverage.json")
    )

    contrast_output = "\n\nCONTRAST ENCODING:\n"
    for i, name in enumerate(contrast_names):
        contrast_output = contrast_output + "c{}: {}\n".format(i, name)

    # Move to output module
    with open(os.path.join(output_dir, "SIE_binomial_reg.txt"), "w") as f:
        f.write(model_summary.as_text())
        f.write("\n\n")
        f.write(contrasts_summary.as_text())
        f.write(contrast_output)

    return results
