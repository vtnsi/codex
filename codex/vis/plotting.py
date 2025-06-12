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
import numpy as np
import statsmodels.api as sm
from scipy.stats import pearsonr
import logging
import matplotlib.patches as patches

from output import results, output
from vis import maps, metrics

# MAIN PLOTTING ENTRY POINTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

savefig = True


def coverage_map(
    map_var, coverage_results, t, output_dir, coverage_subset=None, **kwargs
):
    if "direction" in kwargs:
        counts = coverage_results["results"][t]["sdcc counts"]
    else:
        print(map_var, f"t={t}")
        output.output_json_readable(coverage_results, print_json=True)
        print(f"!t={t}", coverage_results["results"][t].keys())
        counts = coverage_results["results"][t]["combination counts"]
    combination_names = coverage_results["results"][t]["combinations"]

    square = maps.map_info_data(map_var=map_var, counts=counts, kwargs=kwargs)
    vmin, vmax, cmap, cbar_kws = maps.map_info_var(map_var, kwargs=kwargs)
    title, filename = maps.map_info_txtl(
        map_var, coverage_results, t, coverage_subset, kwargs=kwargs
    )

    plt.figure(figsize=(10, 8))
    plt.tight_layout(pad=1.5)
    heatmap = sns.heatmap(
        square,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        # cbar_kws=cbar_kws,
        yticklabels=combination_names,
        linewidths=0.5,
        linecolor="black",
    )
    heatmap.tick_params(axis="both", which="both", length=0, rotation=0)
    size = square.shape
    heatmap.add_patch(
        patches.Rectangle(
            [0, 0],
            width=size[1],
            height=size[0],
            linewidth=3,
            edgecolor="black",
            fill=False,
        )
    )

    cmap.set_under("w")
    cmap.set_over("k")
    cbar = heatmap.collections[0].colorbar
    maps.set_colorbar(cbar, cbar_kws)
    maps.set_plot_text(title)

    maps.save_plots(output_dir, filename, svg=True)

    plt.clf()
    return heatmap


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


def __extract_rank_samples__(i, counts, perf, human_readable):
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
            x.append(metrics.standardized_proportion_per_interaction(counts, i, j))
            y.append(y_raw[j])

    return x, y, z


def __sort_slope_indices__(counts, perf, human_readable):
    slopes = []
    undef_idx = []
    signif_idx = []

    for i in range(len(counts)):
        x, y, z = __extract_rank_samples__(i, counts, perf, human_readable)
        x = np.array(x)
        y = np.array(y)

        p = sns.regplot(x=x, y=y)
        px = p.get_lines()[i].get_xdata()
        py = p.get_lines()[i].get_ydata()

        # MAKE VERSION WITH STATISTICALLY SIGNIFICANT, MARK ON THE LEGEND
        try:
            m = (py[-1] - py[0]) / (px[-1] - px[0])
            # b = py[0]-(m*px[0])

            T_test = metrics.slope_test(x, y)
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


def __plot_regression(
    subset,
    counts,
    perf,
    human_readable,
    p_lines,
    ranks,
    save_per_interaction,
    axis_size,
    labelpad,
    metric,
    outputPath,
):
    num_combinations = len(counts)

    # Slopes
    for i in range(num_combinations):
        if i not in subset:
            continue

        x, y, z = __extract_rank_samples__(i, counts, perf, human_readable)
        px = p_lines[i].get_xdata()
        py = p_lines[i].get_ydata()

        perf_range = np.max(y) - np.min(y)
        perf_var = np.square(np.std(y))

        no_slope = False
        try:
            T_test = metrics.slope_test(x, y)
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
            plt.ylabel(
                "Metric: {}".format(metric), fontsize=axis_size, labelpad=labelpad
            )

            plt.grid(visible=True, axis="y")
            plt.legend()
            filename = "pxi_performance_vs_freq-{}.png".format(ranks[i])
            plt.savefig(os.path.join(outputPath, filename))
            plt.clf()


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
    bound_lim=True,
    save_per_interaction=False,
):
    title_size = 18
    titlepad = 16
    labelpad = 12
    axis_size = int(4 * title_size / 5)

    p, ordered_idx, signif_idx = __sort_slope_indices__(counts, perf, human_readable)

    p_lines = p.get_lines()
    figsize_custom = (11, 8)

    plt.clf()
    if mode == "highlights":
        plt.figure(figsize=figsize_custom)
        subset = ordered_idx
        plt.title(
            textwrap.fill(
                "Standardized Proportion of Interaction Frequency against Performance for {}, subset of interactions | t={}".format(
                    name, str(t)
                ),
                60,
            ),
            pad=titlepad + 3,
            weight="bold",
            fontsize=title_size,
        )
        filename = "pxi_performance_vs_freq-{}_subset.png".format(t)
    elif mode == "signif":
        plt.figure(figsize=figsize_custom)
        subset = signif_idx
        # print("NUMBER OF SIGNIFICANT COMBOS", len(signif_idx))
        plt.title(
            textwrap.fill(
                "Standardized Proportion of Interaction Frequency against Performance for {}, Statistically Signifcant Regression | t={}".format(
                    name, str(t)
                ),
                60,
            ),
            pad=titlepad + 3,
            weight="bold",
            fontsize=title_size,
        )
        filename = "pxi_performance_vs_freq-{}_signif.png".format(t)
    elif mode == "all":
        plt.figure(figsize=figsize_custom)
        plt.title(
            textwrap.fill(
                "Standardized Proportion of Interaction Frequency against Performance for {} | t={}".format(
                    name, str(t)
                ),
                60,
            ),
            pad=titlepad,
            weight="bold",
            fontsize=title_size,
        )
        subset = range(len(counts))
        filename = "pxi_performance_vs_freq-{}_ALL.png".format(t)

    # for all c_l's, min lower bound and max upper bound
    num_combinations = len(counts)
    __plot_regression(
        subset,
        counts,
        perf,
        human_readable,
        p_lines,
        ranks,
        save_per_interaction,
        axis_size,
        labelpad,
        metric,
        outputPath,
    )

    N = int(np.sum(counts[0]))
    print("T", t)
    lower, upper = metrics.standardized_proportion_frequency_bounds_iterative(N, counts)

    """plt.xlabel(
        "Standardized proportion, " + r"$\frac{n_{jl}-\frac{N}{c_j}}{N}$",
        fontsize=axis_size,
        labelpad=labelpad,
    )
    plt.ylabel("Metric: {}".format(metric), fontsize=axis_size, labelpad=labelpad)"""
    # plt.tick_params(fontsize=10)
    # plt.ticklabel_format(font)

    # bound_lim = T
    vlines = False
    if vlines:
        plt.vlines(lower, ymin=0.5, ymax=1)
        plt.vlines(upper, ymin=0.5, ymax=1)
    if bound_lim:
        plt.xlim((lower - 0.05, upper + 0.05))
        plt.ylim((0.5, 1))
    else:
        plt.ylim((0, 1))
        plt.xlim((-1, 1))

    cmap = sns.diverging_palette(145, 300, s=60, as_cmap=True)

    plt.vlines(0.0, ymin=0, ymax=1, color="black", linestyles=["dashed"])
    plt.fill_betweenx(
        range(5),
        -2,
        0,
        color=cmap(55),
        alpha=0.18,
        label="Underrepresented interaction",
    )
    plt.fill_betweenx(
        range(5),
        0,
        2,
        color=cmap(225),
        alpha=0.18,
        label="Overrepresentation interaction",
    )

    plt.tight_layout(pad=4)
    plt.xlabel(
        "Standardized Proportion Frequency Coverage",
        fontsize=axis_size,
        labelpad=labelpad,
    )
    plt.ylabel(
        "Per-interaction Performance: predicting on {}".format(metric),
        fontsize=axis_size,
        labelpad=labelpad,
    )
    plt.grid(visible=True, axis="y")
    plt.legend()

    if savefig:
        plt.savefig(os.path.join(outputPath, filename))

    plt.cla()
    return


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
