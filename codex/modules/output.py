# Author Brian Lee leebri2n@vt.edu
# Created Date: Mar 1 2023
# Updated Date: 3 April, 2024
# Movement to new repo: 12 Apr, 2024

import sys
import os
import json
import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import math
from datetime import datetime
import copy
import scipy
import logging
import glob

from vis import plotting, maps, metrics
from utils import results

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


def dataset_eval_vis(
    output_dir, dataset_name, coverage_results, strengths, funct=[math.log10]
):
    output_dir = make_output_dir_nonexist(os.path.join(output_dir, "CC"))
    for t in strengths:
        counts = coverage_results[t]["combination counts"]
        ranks = coverage_results[t]["combinations"]
        cc = coverage_results[t]["CC"]
        cc = round(cc, 4)

        LOGGER_OUT.log(
            level=25,
            msg="t = {}\n{}".format(t, [coverage_results[t]["CC"] for t in strengths]),
        )

        create_coverage_map(
            "binary", output_dir, dataset_name, t, counts, ranks, cc_value=cc
        )
        create_coverage_map("frequency", output_dir, dataset_name, t, counts, ranks)
        create_coverage_map(
            "function_frequency",
            output_dir,
            dataset_name,
            t,
            counts,
            ranks,
            funct=funct,
        )
        create_coverage_map(
            "proportion_frequency", output_dir, dataset_name, t, counts, ranks
        )
        create_coverage_map(
            "proportion_frequency_standardized",
            output_dir,
            dataset_name,
            t,
            counts,
            ranks,
        )
    plt.close("all")

    return output_json_readable(
        coverage_results,
        write_json=True,
        file_path=os.path.join(output_dir, "coverage.json"),
    )


def dataset_split_eval_vis(
    output_dir,
    dataset_name,
    coverage_results,
    strengths,
    split_id,
    funct=[math.log10],
    comparison=False,
):
    if comparison:
        output_dir = make_output_dir_nonexist(os.path.join(output_dir, split_id))
    cc_output_dir = make_output_dir_nonexist(os.path.join(output_dir, "CC"))
    sdcc_output_dir = make_output_dir_nonexist(os.path.join(output_dir, "SDCC"))

    for t in strengths:
        sdcc_directions = [
            key for key in coverage_results[t] if "-" in key and "val" not in key
        ]
        for direction in sdcc_directions:
            target_name = direction.split("-")[0]
            source_name = direction.split("-")[1]
            sourceCC = coverage_results[t][source_name]
            targetCC = coverage_results[t][target_name]
            source_ranks = coverage_results[t][direction]["combinations"]
            target_ranks = coverage_results[t][direction]["combinations"]

            source_cc = coverage_results[t][source_name]["CC"]
            target_cc = coverage_results[t][target_name]["CC"]
            sdcc = coverage_results[t][direction]["SDCC"]

            source_cc = round(source_cc, 4)
            target_cc = round(target_cc, 4)
            sdcc = round(sdcc, 4)

            # SOURCE
            create_coverage_map(
                "binary",
                cc_output_dir,
                dataset_name,
                t,
                sourceCC["combination counts"],
                source_ranks,
                cc_value=source_cc,
                subset_name=source_name,
            )
            create_coverage_map(
                "frequency",
                cc_output_dir,
                dataset_name,
                t,
                sourceCC["combination counts"],
                source_ranks,
                cc_value=source_cc,
                subset_name=source_name,
            )
            create_coverage_map(
                "function_frequency",
                cc_output_dir,
                dataset_name,
                t,
                sourceCC["combination counts"],
                source_ranks,
                funct=funct,
                cc_value=source_cc,
                subset_name=source_name,
            )
            create_coverage_map(
                "proportion_frequency",
                cc_output_dir,
                dataset_name,
                t,
                sourceCC["combination counts"],
                source_ranks,
                cc_value=source_cc,
                subset_name=source_name,
            )
            create_coverage_map(
                "proportion_frequency_standardized",
                cc_output_dir,
                dataset_name,
                t,
                sourceCC["combination counts"],
                source_ranks,
                cc_value=source_cc,
                subset_name=source_name,
            )

            # TARGET
            create_coverage_map(
                "binary",
                cc_output_dir,
                dataset_name,
                t,
                targetCC["combination counts"],
                target_ranks,
                cc_value=target_cc,
                subset_name=target_name,
            )
            create_coverage_map(
                "frequency",
                cc_output_dir,
                dataset_name,
                t,
                targetCC["combination counts"],
                target_ranks,
                cc_value=target_cc,
                subset_name=target_name,
            )
            create_coverage_map(
                "function_frequency",
                cc_output_dir,
                dataset_name,
                t,
                targetCC["combination counts"],
                target_ranks,
                funct=funct,
                cc_value=target_cc,
                subset_name=target_name,
            )
            create_coverage_map(
                "proportion_frequency",
                cc_output_dir,
                dataset_name,
                t,
                targetCC["combination counts"],
                target_ranks,
                cc_value=target_cc,
                subset_name=target_name,
            )
            create_coverage_map(
                "proportion_frequency_standardized",
                cc_output_dir,
                dataset_name,
                t,
                targetCC["combination counts"],
                target_ranks,
                cc_value=target_cc,
                subset_name=target_name,
            )

            SDCCconstraints = coverage_results[t][direction]
            assert source_ranks == target_ranks
            create_coverage_map(
                "sdcc_binary_constraints",
                sdcc_output_dir,
                dataset_name,
                t,
                SDCCconstraints["sdcc counts"],
                source_ranks,
                source_name=source_name,
                target_name=target_name,
                sdcc_value=sdcc,
            )
            """create_coverage_map(
                "sdcc_binary_constraints_neither",
                sdcc_output_dir,
                dataset_name,
                t,
                SDCCconstraints["sdcc counts"],
                source_ranks,
                source_name=source_name,
                target_name=target_name,
                sdcc_value=sdcc,
            )"""

        plotting.split_elements_bar(
            output_dir, dataset_name, coverage_results, t, split_id, sdcc_directions
        )
    plt.close("all")

    return output_json_readable(
        coverage_results,
        write_json=True,
        file_path=os.path.join(output_dir, "coverage.json"),
    )


def dataset_split_comp_vis(
    output_dir,
    dataset_name,
    coverage_multi,
    performance_multi,
    strengths,
    metric,
    split_ids=None,
):
    output_dir_outer = make_output_dir_nonexist(output_dir)

    for split_id in split_ids:
        # for split in coverage_multi:
        cc_output_dir = make_output_dir_nonexist(
            os.path.join(output_dir_outer, split_id, "CC")
        )
        sdcc_output_dir = make_output_dir_nonexist(
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


def performance_by_interaction_vis(
    output_dir,
    dataset_name,
    coverage_results,
    strengths,
    metric,
    order,
    display_n,
    subset,
    display_all_lr=False,
):
    cc_output_dir = make_output_dir_nonexist(os.path.join(output_dir, "CC"))
    for t in strengths:
        counts = coverage_results[t]["combination counts"]
        ranks = coverage_results[t]["combinations"]
        perf = coverage_results[t]["performance"][metric]

        create_coverage_map(
            "binary", cc_output_dir, dataset_name, t, counts, ranks, subset
        )
        create_coverage_map(
            "frequency", cc_output_dir, dataset_name, t, counts, ranks, subset
        )
        create_coverage_map(
            "proportion_frequency",
            cc_output_dir,
            dataset_name,
            t,
            counts,
            ranks,
            subset,
        )
        create_coverage_map(
            "proportion_frequency_standardized",
            cc_output_dir,
            dataset_name,
            t,
            counts,
            ranks,
            subset,
        )
        create_coverage_map(
            "performance",
            cc_output_dir,
            dataset_name,
            t,
            counts,
            ranks,
            subset,
            performance_all_interactions=perf,
            metric=metric,
        )

        interactions_consolidated = results.consolidated_interaction_info(
            coverage_results, strengths, metric, order=order, display_n=display_n
        )
        plotting.plot_pbi_bar(
            output_dir, interactions_consolidated, t, metric, display_n, order
        )

        human_readable = coverage_results[t]["human readable performance"][metric]
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
        )

    coverage_results = output_json_readable(
        coverage_results,
        write_json=True,
        file_path=os.path.join(output_dir, "coverage.json"),
    )
    plt.close("all")
    return coverage_results


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
    output_dir = make_output_dir_nonexist(output_dir)
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


# -------------------- Coverage maps -------------------- #


def create_coverage_map(
    mode,
    output_dir,
    dataset_name,
    t,
    counts,
    rank_labels,
    cc_value=None,
    sdcc_value=None,
    subset_name=None,
    **kwargs,
):
    """
    Creates square heatmaps for coverage visualization.

    Modes:
    - frequency: Raw frequency of interactions. Scaled from fewest to greatest.
    - function_frequency: Function applied to raw frequency. Scaled from funct(fewest) to funct(greatest).
    - proportion_frequency: Proportion of an interaction's appearences for a given combination. Intended
        to measure dominance of an interaction within a combination of features. Scaled from 0-1.
    - performance: Per-interaction performance drawn from per-sample performance. Scaled from 0-1.
        Requires calculated per-sample performance file.
    - binary: True/False appearence of an interaction. Values are 0 or 1.
    """
    boxsize = max(len(x) for x in counts)
    square = np.full([len(counts), boxsize], dtype=float, fill_value=-1)
    filename = None
    cmap = None
    cbar_kws = None
    cbar_ticklabels = None
    vmax = None
    vmin = 0
    xlabel = None
    if subset_name is None:
        subset_name = "all"

    if mode == "frequency":
        square, cmap, cbar_kws, cbar_ticklabels = maps.frequency_map(square, counts)
        cmap.set_under("w")

        filename = "CC_frequency-t{}_{}_{}.png".format(t, dataset_name, subset_name)
        title = "{}-way Frequency Coverage of {}".format(t, dataset_name)

    elif mode == "function_frequency":
        funct = kwargs["funct"]

        assert type(funct) is list
        for i, function in enumerate(funct):
            funct_name = str(function)

            square, cmap, cbar_kws = maps.frequency_map_function(
                square, counts, function
            )
            cmap.set_under("w")

            filename = "CC_function_frequency-{}-t{}_{}_{}.png".format(
                funct_name, t, dataset_name, subset_name
            )
            title = "{}-way {} Frequency Coverage of {} over".format(
                t, funct_name, dataset_name
            )
        return

    elif mode == "proportion_frequency":
        square, cmap, cbar_kws, cbar_ticklabels = maps.frequency_map_proportion(
            square, counts
        )
        # lim = np.max([np.abs(np.min(square)), np.abs(np.max(square))])
        vmin = 0
        vmax = 1
        filename = "CC_frequency_proportion-t{}_{}_{}.png".format(
            t, dataset_name, subset_name
        )
        title = "{}-way Proportion Frequency Coverage of {}".format(t, dataset_name)

    elif mode == "proportion_frequency_standardized":
        c_is = [len(counts_one_combo) for counts_one_combo in counts]
        vmin = -1  # round(0 - (1/np.min(c_is)), 4)
        vmax = 1  # round(1 - (1/np.max(c_is)))

        # midpoint = np.mean([vmin, vmax])

        square = np.full([len(counts), boxsize], dtype=float, fill_value=-999)
        square, cmap, cbar_kws, cbar_ticklabels = (
            maps.frequency_map_proportion_standardized(square, counts)
        )
        cmap.set_under("w")

        filename = "CC_frequency_proportion_standardized-t{}_{}_{}.png".format(
            t, dataset_name, subset_name
        )
        title = "{}-way Standardized Proportion Frequency Coverage of {}".format(
            t, dataset_name
        )
        # "Interactions"#, " + r"$\frac{n_{jl}-\frac{N}{c_j}}{N}$"
        xlabel = None

    elif mode == "performance":
        perf = kwargs["performance_all_interactions"]
        metric = kwargs["metric"]
        square, cmap, cbar_kws, cbar_ticklabels = maps.performance_map(
            square, counts, perf=perf, metric=metric
        )
        vmin = 0
        vmax = 1

        filename = "CC_pi_performance-t{}_{}_{}.png".format(
            t, dataset_name, subset_name
        )
        title = "Performance, {}, per {}-way Interactions in {}".format(
            metric, t, dataset_name
        )

    elif mode == "binary":
        # Takes in a list of lists that represents interaction coverage
        # produces a map and colors the times an interaction appears
        # 0 = not covered, x > 0 --> covered x times, -1 indicates invalid interaction
        # one list per rank, but rank lists have different size
        # one row per list, possibly with white space

        if dataset_name == "testD":
            dataset_name = "$test_D$"

        square, cmap, cbar_kws, cbar_ticklabels = maps.frequency_map_binary(
            square, counts
        )

        filename = "CC_binary-t{}_{}_{}.png".format(t, dataset_name, subset_name)
        title = "{}-way Binary Coverage for {} over {} (CC={})".format(
            t, dataset_name, subset_name, cc_value
        )

    elif mode == "sdcc_binary_constraints":
        square_sdcc = np.full([len(counts), boxsize], dtype=float, fill_value=-2)

        source_name = kwargs["source_name"]
        target_name = kwargs["target_name"]
        vmin = -1
        vmax = 1

        square, cmap, cbar_kws, cbar_ticklabels = maps.sd_map_binary_constrained(
            square_sdcc, counts
        )

        title = "{}-way Coverage in {} not appearing in {}, {} (SDCC={})".format(
            t, target_name, source_name, dataset_name, sdcc_value
        )
        filename = "SDCC-t{}-way Set Diff {} not appearing in {}_{}.png".format(
            t, target_name, source_name, dataset_name
        )

    elif mode == "sdcc_binary_constraints_neither":
        square_sdcc = np.full([len(counts), boxsize], dtype=float, fill_value=-2)

        source_name = kwargs["source_name"]
        target_name = kwargs["target_name"]

        if target_name == "testC":
            target_name = "$test_C$"
        if target_name == "testD":
            target_name = "$test_D$"
        if source_name == "trainC":
            source_name = "$train_C$"

        vmin = -1
        vmax = 2

        square, cmap, cbar_kws, cbar_ticklabels = (
            maps.sd_map_binary_constrained_neither(
                square_sdcc, counts, source_name=source_name
            )
        )
        title = "{}-way Coverage in {} not appearing in {}, {} (SDCC={})".format(
            t, target_name, source_name, dataset_name, sdcc_value
        )
        filename = (
            "SDCC-t{}-way Set Diff {} not appearing in {}_{}_wneither.png".format(
                t, target_name, source_name, dataset_name
            )
        )

    else:
        raise NameError("No coverage map scheme found! For mode: <{}>".format(mode))

    # HEATMAP CALL
    plotting.heatmap(
        output_dir,
        square,
        vmin,
        vmax,
        cmap,
        cbar_kws,
        rank_labels,
        title,
        filename,
        xlabel=xlabel,
        cbar_ticklabels=cbar_ticklabels,
        mode=mode,
    )
    filename_svg = "{}.svg".format(filename.split(".")[0])
    plotting.heatmap(
        output_dir,
        square,
        vmin,
        vmax,
        cmap,
        cbar_kws,
        rank_labels,
        title,
        filename_svg,
        xlabel=xlabel,
        cbar_ticklabels=cbar_ticklabels,
        mode=mode,
    )

    plt.clf()
    plt.close("all")
    return


# -------------------------- WRITING -------------------------- #
# writes the CC for a t to the t file


def writeCCtToFile(output_dir, name, t, CC):
    output_dir = make_output_dir_nonexist(os.path.join(output_dir, "CC"))
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
    output_dir = make_output_dir_nonexist(os.path.join(output_dir, "SDCC"))
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
    output_dir = make_output_dir_nonexist(os.path.join(output_dir, "CC"))
    filename = os.path.join(output_dir, "t{}_{}_missing.txt".format(t, name))

    with open(filename, mode="w") as tfile:
        tfile.write("Missing interactions for strength " + str(t) + ":\n")
        for interaction in decodedMissing:
            tfile.write(str(interaction) + "\n")
        tfile.close()


def writeSetDifferencetoFile(
    output_dir, sourcename, targetname, t, setDifferenceInteractions
):
    output_dir = make_output_dir_nonexist(os.path.join(output_dir, "SDCC"))
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


def make_output_dir_nonexist(output_dir_new, timed=False):
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
    return output_dir_new


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
