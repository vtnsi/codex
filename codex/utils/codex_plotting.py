import statsmodels.regression.linear_model as sm_lin
import sys
import os
import json
import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import scipy.stats
import seaborn as sns
import textwrap
import math
from datetime import datetime
from scipy.interpolate import UnivariateSpline, interp1d
import copy
import scipy
import statsmodels.api as sm
from scipy.stats import pearsonr
import logging

from utils import codex_metrics, results_handler

# MAIN PLOTTING ENTRY POINTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

savefig = True


def plot_probe_exploit_suite(
    output_dir,
    interactions_probe,
    interactions_exploit,
    t,
    separate=False,
    savefig=True,
):
    gridspec = {"width_ratios": [3, 1], "height_ratios": [1, 1]}
    fig, ax = plt.subplots(2, 2, gridspec_kw=gridspec, figsize=(12, 10))
    fig.suptitle("Model Probing", weight="bold", fontsize=24)

    # t = str(t)
    width = 0.4

    bottom_common, bottom_probe_perf, bottom_exploit_perf = (
        results_handler.consolidated_interaction_info(
            interactions_probe, interactions_exploit, t
        )
    )
    bottom_common = [textwrap.fill(string, 12) for string in bottom_common]
    idx = np.arange(len(bottom_common))

    plt.subplot(2, 2, 1)
    plt.title("Performance by interaction, Probe vs. Exploit, t = {}".format(t))
    plt.tight_layout()
    plt.bar(idx - 0.2, bottom_probe_perf, width)
    plt.bar(idx + 0.2, bottom_exploit_perf, width)

    plt.ylabel("Precision")
    plt.ylim(0, 1)
    plt.xlabel("Interactions")
    plt.xticks(ticks=idx, labels=bottom_common, rotation=20)
    plt.grid(visible=True, axis="y")

    plt.subplot(2, 2, 2)
    plt.title("Interactions in probe set", pad=32, weight="bold")
    # plt.tight_layout()
    """output_json_readable(interactions_probe, print_json=True)
    for interaction in interactions_probe[t]:
        if interaction != 'top interactions' and interaction != 'bottom interactions':
            print(interactions_probe[t][interaction]['counts'])
    exit()"""
    sizes = [
        interactions_probe[t][interaction]["counts"]
        for interaction in interactions_probe[t]
        if interaction != "bottom interactions" and interaction != "top interactions"
    ]
    labels = [
        textwrap.fill(interaction, 12)
        for interaction in interactions_probe[t]
        if interaction != "bottom interactions" and interaction != "top interactions"
    ]
    if len(labels) >= 10:
        labels = None
    plt.pie(sizes, labels=labels)

    plt.subplot(2, 2, 4)
    plt.title("Interactions in exploit set", pad=32, weight="bold")
    # plt.tight_layout()

    sizes_ex = [
        interactions_exploit[t][interaction]["counts"]
        for interaction in interactions_exploit[t]
        if interaction != "bottom interactions" and interaction != "top interactions"
    ]
    labels_ex = [
        textwrap.fill(interaction, 12)
        for interaction in interactions_exploit[t]
        if interaction != "bottom interactions" and interaction != "top interactions"
    ]
    if len(labels_ex) >= 10:
        labels_ex = None
    plt.pie(sizes_ex, labels=labels_ex)

    plt.subplot(2, 2, 3)
    plt.title("Distribution of per-interaction performance")
    plt.tight_layout()
    performances = [
        interactions_probe[t][interaction]["performance"]
        for interaction in interactions_probe[t]
        if interaction != "bottom interactions" and interaction != "top interactions"
    ]
    performances_ex = [
        interactions_exploit[t][interaction]["performance"]
        for interaction in interactions_exploit[t]
        if interaction != "bottom interactions" and interaction != "top interactions"
    ]
    performances_all = performances + performances_ex
    plt.hist(performances_all, color="Purple")
    plt.xlim(0, max(performances_all))
    plt.xlabel("Performance")

    if savefig:
        plt.savefig(
            os.path.join(output_dir, "model_probing_SUITE-t{}.png".format(str(t)))
        )

    plt.clf()
    return


def split_comp_scatter(
    output_dir,
    dataset_name,
    coverages: dict,
    performances: dict,
    t,
    direction,
    metric,
    split_ids=None,
    savefig=True,
    showfig=False,
):
    if "-" not in direction:
        coverage_metric = "CC"
    else:
        coverage_metric = "SDCC"

    coverages_sorted = dict(sorted(coverages.items()))
    performances_sorted = dict(sorted(performances.items()))

    fig, ax = plt.subplots(1, 1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("t={}".format(t))
    plt.grid(visible=True)
    ids = list(coverages_sorted.keys())
    filename = "{}-dataset_split_comparison_{}.png".format(direction, t)

    for split_id in split_ids:
        x_single = coverages_sorted[split_id][t][direction][coverage_metric]
        y_single = performances[split_id]["test"]["Overall Performance"][metric]
        plt.scatter(x_single, y_single, label=split_id)

        plt.tight_layout()

    plt.title("Dataset SDCC Comparison against Performance")
    plt.grid(visible=True, which="both")
    plt.legend()
    plt.xlabel("{} {} value".format(direction, coverage_metric))
    plt.ylabel("Performance: {}".format(metric))

    if savefig:
        plt.savefig(os.path.join(output_dir, filename))
    if showfig:
        plt.show()
    plt.clf()
    return


def extract_rank_samples(i, counts, perf, human_readable):
    x = []
    y = []
    z = []
    x_raw = counts[i]
    # For intteraction j in rank i
    for j in range(len(x_raw)):
        y_raw = perf[i]
        if list(human_readable[i].keys()) not in z:
            z.append(list(human_readable[i].keys()))

        if y_raw[j] is None:
            continue
        else:
            x.append(
                codex_metrics.standardized_proportion_per_interaction(counts, i, j)
            )
            y.append(y_raw[j])

    return x, y, z


def sort_slope_indices(counts, perf, human_readable):
    slopes = []
    undef_idx = []
    signif_idx = []

    for i in range(len(counts)):
        x, y, z = extract_rank_samples(i, counts, perf, human_readable)
        x = np.array(x)
        y = np.array(y)

        p = sns.regplot(x=x, y=y)
        px = p.get_lines()[i].get_xdata()
        py = p.get_lines()[i].get_ydata()

        # MAKE VERSION WITH STATISTICALLY SIGNIFICANT, MARK ON THE LEGEND
        try:
            m = (py[-1] - py[0]) / (px[-1] - px[0])
            # b = py[0]-(m*px[0])

            T_test = codex_metrics.slope_test(x, y)
            """print(T_test.pvalue[1])"""
            if T_test.pvalue[1] <= 0.05:
                signif_idx.append(i)

        except:
            m = np.NaN

            undef_idx.append(i)
            print("No slope for feature {}".format(i))

        slopes.append(m)

    order = np.argsort(slopes)
    ordered_idx = list(order[: len(order) - len(undef_idx)])

    if len(ordered_idx) > 5:
        ordered_idx = (
            ordered_idx[:2]
            + [ordered_idx[int(len(ordered_idx) / 2)]]
            + ordered_idx[-2:]
        )
    return p, ordered_idx, signif_idx


def plot_pbi_frequency_scatter(
    mode,
    outputPath,
    name,
    counts,
    perf,
    human_readable,
    t,
    metric,
    ranks,
    display_all_lr: bool,
    save_per_interaction=False,
):
    title_size = 18
    titlepad = 16
    labelpad = 12
    axis_size = int(4 * title_size / 5)

    p, ordered_idx, signif_idx = sort_slope_indices(counts, perf, human_readable)
    p_lines = p.get_lines()

    plt.clf()
    if mode == "highlights":
        plt.figure(figsize=(14, 10))
        subset = ordered_idx
        plt.title(
            textwrap.fill(
                "Standardized Proportion of Interaction Frequency against Performance for {}, Subset | t={}".format(
                    name, str(t)
                ),
                54,
            ),
            pad=titlepad,
            weight="bold",
            fontsize=title_size,
        )
        filename = "pxi_performance_vs_freq-{}_subset.png".format(t)
    elif mode == "signif":
        plt.figure(figsize=(14, 10))
        subset = signif_idx
        plt.title(
            textwrap.fill(
                "Standardized Proportion of Interaction Frequency against Performance for {}, Statistically Signifcant Regression | t={}".format(
                    name, str(t)
                ),
                54,
            ),
            pad=titlepad,
            weight="bold",
            fontsize=title_size,
        )
        filename = "pxi_performance_vs_freq-{}_signif.png".format(t)
    elif mode == "all":
        plt.figure(figsize=(10, 15))
        plt.tight_layout()
        plt.title(
            textwrap.fill(
                "Standardized Proportion of Interaction Frequency against Performance for {} | t={}".format(
                    name, str(t)
                ),
                54,
            ),
            pad=titlepad,
            weight="bold",
            fontsize=title_size,
        )
        subset = range(len(counts))
        filename = "pxi_performance_vs_freq-{}_ALL.png".format(t)

    c_l = len(counts)
    counts_array = np.array(counts, dtype=object)

    N = np.sum(counts_array)
    for i in range(c_l):
        if i not in subset:
            continue

        x, y, z = extract_rank_samples(i, counts, perf, human_readable)
        px = p_lines[i].get_xdata()
        py = p_lines[i].get_ydata()

        perf_range = np.max(y) - np.min(y)
        perf_var = np.square(np.std(y))

        no_slope = False
        try:
            T_test = codex_metrics.slope_test(x, y)
            p_value_m = T_test.pvalue[1]
        except:
            no_slope = True
            p_value_m = np.NaN

        try:
            m = (py[-1] - py[0]) / (px[-1] - px[0])
            b = py[0] - (m * px[0])
        except:
            m = np.NaN
            b = np.NaN

        text_str = text_str = (
            "\n     "
            + r"$slope_{reg}$: "
            + str(round(m, 3))
            + r", $P-val: $"
            + str(round(p_value_m, 3))
            + "\n     "
            + r"$range_{perf}$: "
            + str(round(perf_range, 3))
            + ", "
            + r"$\sigma^2_{perf}$: "
            + str(round(perf_var, 3))
        )
        plt.scatter(x, y, label=ranks[i] + text_str)
        if not no_slope:
            sns.regplot(x=x, y=y)
        if save_per_interaction:
            plt.xlabel(
                "Standardized proportion, " + r"$\frac{n_{jl}-\frac{N}{c_j}}{N}$",
                fontsize=axis_size,
                labelpad=labelpad,
            )
            plt.ylabel("Metric: {}".format(metric), fontsize=axis_size, labelpad=labelpad)
            plt.ylim((0, 1))
            lower, upper = codex_metrics.standardized_proportion_frequency_bounds(N, c_l)
            plt.xlim((-1,1))
            plt.grid(visible=True, axis="y")
            plt.legend()
            filename = 'pxi_performance_vs_freq-{}.png'.format(ranks[i])
            plt.savefig(os.path.join(outputPath, filename))
            plt.clf()

    plt.xlabel(
        "Standardized proportion, " + r"$\frac{n_{jl}-\frac{N}{c_j}}{N}$",
        fontsize=axis_size,
        labelpad=labelpad,
    )
    plt.ylabel("Metric: {}".format(metric), fontsize=axis_size, labelpad=labelpad)
    plt.ylim((0, 1))
    plt.xlim((-1,1))
    plt.grid(visible=True, axis="y")
    plt.legend()

    if savefig:
        plt.savefig(os.path.join(outputPath, filename))


def plot_pbi_bar(
    output_dir, interactions, t: str, metric, display_n, order, savefig=True
):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    width = 0.4

    top = interactions[t]["top interactions"][:display_n]
    top_perf = [interactions[t][interaction]["performance"] for interaction in top][
        :display_n
    ]
    bottom = interactions[t]["bottom interactions"][:display_n]
    bottom_perf = [
        interactions[t][interaction]["performance"] for interaction in bottom
    ][:display_n]
    top = [textwrap.fill(string, 12) for string in top][:display_n]
    bottom = [textwrap.fill(string, 12) for string in bottom][:display_n]
    idx = np.arange(len(bottom))

    plt.subplot(2, 1, 1)
    plt.title("Performance by Interaction, Bottom Interactions")
    # plt.tight_layout()
    plt.bar(idx, bottom_perf, width, color="red")
    plt.ylabel(metric)
    plt.ylim(0, 1)
    plt.xlabel("Bottom Interactions")
    plt.xticks(ticks=idx, labels=bottom, rotation=20)
    plt.grid(visible=True, axis="y")

    plt.subplot(2, 1, 2)
    plt.title("Performance by Interaction, Top Interactions")
    # plt.tight_layout()
    plt.bar(idx, top_perf, width, color="limegreen")
    plt.ylabel(metric)
    plt.ylim(0, 1)
    plt.xlabel("Top Interactions")
    plt.xticks(ticks=idx, labels=top, rotation=20)
    plt.grid(visible=True, axis="y")

    if savefig:
        plt.savefig(os.path.join(output_dir, "pxi_performing-t{}.png".format(t)))

    plt.clf()
    return interactions


def heatmap(
    output_dir,
    square,
    vmin,
    vmax,
    cmap,
    cbar_kws,
    rank_labels,
    title,
    filename,
    xlabel=None,
    ylabel=None,
    cbar_ticklabels=None,
    mode=None,
    outlines=True
):
    if outlines:
        linewidths=0.5
    else:
        linewidths=0
    plt.figure(figsize=(12, 7.5))
    heatmap = sns.heatmap(
        square,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        cbar_kws=cbar_kws,
        yticklabels=rank_labels,
        linewidths=linewidths,
        linecolor="black",
    )
    heatmap.tick_params(axis="both", which="both", length=0)

    title = textwrap.fill(title, 40)

    cmap.set_under("w")
    # cmap.set_over('k')
    colorbar = heatmap.collections[0].colorbar

    if mode == "sdcc_binary_constraints_neither":
        colorbar.set_ticks([-0.667, 0, 0.667, 1.667])
        colorbar.set_ticklabels(cbar_ticklabels, rotation=-30)
    elif mode == "sdcc_binary_constraints":
        colorbar.set_ticks([-0.667, 0, 0.667])
        colorbar.set_ticklabels(cbar_ticklabels, rotation=-30)
    elif mode == "binary":
        colorbar.set_ticks([0, 1])
        colorbar.set_ticklabels(cbar_ticklabels, rotation=-30)

    plt.title(title, weight="bold")
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, labelpad=-5, fontsize=10)
    plt.yticks(rotation=0)
    plt.tight_layout(pad=2)

    # FIGURE SAVE
    with open(os.path.join(output_dir, filename), "wb") as f:
        plt.savefig(f)
    plt.clf()

    return


# Displays the CC per t provided in strength file


def createCCchart(output_dir, name, CC, strength):
    # plot the graph of CC for each t in strength vector
    plt.plot(strength, CC, "o-")
    plt.xlabel("Strength (t)")
    plt.ylabel("Combinatorial Coverage Metric")
    plt.xticks(np.arange(1, len(strength) + 1, step=1))
    title = name + " CC figure"
    plt.title(title)
    filename = output_dir + title + ".png"
    with open(filename, "wb") as figfile:
        plt.savefig(figfile)
    # plt.show()
    plt.clf()


def split_elements_bar(
    output_dir, dataset_name, coverage, t, split_id, sdcc_directions
):
    plt.figure()

    sdcc_val = [coverage[t][direction]["SDCC"] for direction in sdcc_directions]
    assert len(sdcc_directions) == len(sdcc_val)

    idx = np.arange(len(sdcc_directions))

    plt.bar(idx, sdcc_val, width=0.4, color="purple")
    plt.xticks(ticks=idx, labels=sdcc_directions, rotation=20)

    plt.title(
        "SDCC value comparisons for {}, {}, t={}".format(dataset_name, split_id, t)
    )
    plt.ylabel("SDCC: {}".format("SDCC"))
    plt.ylim(0, 1)
    plt.grid(visible=True, axis="y")

    if savefig:
        plt.savefig(
            os.path.join(
                output_dir, "dataset_split_eval_{}-t{}.png".format(split_id, t)
            )
        )
    plt.clf()
    return
