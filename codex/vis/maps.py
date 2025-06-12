import matplotlib.colorbar
import seaborn as sns
import numpy as np
import textwrap
import matplotlib.colors
import matplotlib.pyplot as plt

from vis import metrics

import os

TITLE_SIZE = 18
AXIS_SIZE = int(0.8 * TITLE_SIZE)


def set_plot_text(title, title_size=18):
    plt.gcf()

    title = textwrap.fill(title, 60)
    plt.title(title, weight="bold", fontsize=18, pad=15)

    return


def set_axes(ylab_rotation=0, ylab_alignment="center"):
    plt.gcf()

    plt.xlabel("Interactions", fontsize=AXIS_SIZE, labelpad=15, weight="bold")
    plt.ylabel("Combinations", fontsize=AXIS_SIZE, labelpad=15, weight="bold")
    plt.yticks(fontsize=12, rotation=ylab_rotation, va=ylab_alignment)

    return


def save_plots(output_dir, file_basename, svg=False):
    # savefig
    plt.savefig(os.path.realpath(os.path.join(output_dir, f"{file_basename}.png")))
    if svg:
        plt.savefig(
            os.path.realpath(os.path.join(output_dir, f"{file_basename}.svg")),
            format="svg",
        )


def map_info_data(map_var, counts, kwargs):
    boxsize = max(len(x) for x in counts)
    square = np.full([len(counts), boxsize], dtype=float, fill_value=-1)

    for row in range(len(counts)):
        for col in range(len(counts[row])):
            if counts[row][col] > 0:
                if map_var == "binary":
                    """
                        Fill square map valued on its appearance/absence.
                    """
                    square[row][col] = 1
                elif map_var == "frequency":
                    """
                        Fill square map valued on the raw counts of apperances in the dataset.
                    """
                    square[row][col] = counts[row][col]
                elif map_var == "proportion_frequency":
                    """
                        Fill square map valued on the proportion an interaction makes for
                        a combination's total number of interactions.
                    """
                    square[row][col] = counts[row][col] / np.sum(counts[row])
                    # square[row][col] = 0 if not?
                elif map_var == "proportion_frequency_standardized":
                    """
                        Fill square map valued on the proportion an interaction makes for
                        a combination's total number of interactions, standardized to reflect
                        distance from balance (0).
                    """
                    square[row][col] = metrics.standardized_proportion_per_interaction(
                        counts, row, col
                    )
                    # square[r]w[cio] = -999 if not
                elif map_var == "performance":
                    """
                        Fill square map valued on a model's performance on samples of each t-way 
                        interaction appearing in the dataset.
                    """
                    perf = kwargs["pi_perf"]
                    square[row][col] = perf[kwargs['metric']][row][col]
                elif map_var == "sdcc_binary_constraints":
                    """
                        Takes in a list of lists that represents interaction coverage produces 
                        a map and colors the times an interaction appears 0 = not covered, 
                        x > 0 --> covered x times, -1 indicates invalid interaction one list 
                        per rank, but rank lists have different size one row per list, possibly 
                        with white space.

                        Takes in a list of lists that represents interaction coverage
                        produces a map and colors the times an interaction appears
                        0 = not covered, x > 0 --> covered x times, -1 indicates invalid 
                        interaction (in constraint not in T) one list per rank, but rank 
                        lists have different size one row per list, possibly with white space.
                    """
                    square[row][col] = counts[row][col]
                elif map_var == "sdcc_binary_constraints_wneither":
                    square[row][col] = counts[row][col]
                else:
                    raise NameError

            else:
                assert square[row][col] == -1

    return square


def map_info_txtl(map_var, coverage_results, t, coverage_subset, kwargs):
    """ "
    Mode-specific kwargs:
    - Performance: metric
    """
    try:
        cc_metric = round(coverage_results["results"][t]["CC"], 3)
    except KeyError:
        cc_metric = None

    dataset_name = coverage_results["info"]["dataset_name"]

    if coverage_subset is None:
        subset_specification = ""
        subset_spec_fn = ""
    else:
        subset_specification = f" over {coverage_subset}"
        subset_spec_fn = f"_{coverage_subset}"

    if map_var == "binary":
        title = f"Binary Coverage of {t}-way Interactions in {dataset_name}{subset_specification} | (CC = {cc_metric})"
        file_basename = f"cc_binary_t{t}-{dataset_name}{subset_spec_fn}"

    elif map_var == "frequency":
        title = f"Count Frequency Coverage of {t}-way Interactions in {dataset_name}{subset_specification} | (CC = {cc_metric})"
        file_basename = f"cc_freq_t{t}-{dataset_name}{subset_spec_fn}"

    elif map_var == "proportion_frequency":
        title = f"Proportion Frequency Coverage of {t}-way Interactions in {dataset_name}{subset_specification} | (CC = {cc_metric})"
        file_basename = f"cc_prop_t{t}-{dataset_name}{subset_spec_fn}"

    elif map_var == "proportion_frequency_standardized":
        title = f"Standardized Proportion Frequency Coverage of {t}-way Interactions in {dataset_name}{subset_specification} | (CC = {cc_metric})"
        file_basename = f"cc_stdprop_t{t}-{dataset_name}{subset_spec_fn}"

    elif map_var == "performance":
        metric = kwargs["metric"]
        title = f"Per-Interaction of Performance of {t}-way Interactions in {dataset_name}{subset_specification}, {metric}"
        file_basename = f"perf_{metric}_t{t}-{dataset_name}{subset_spec_fn}"

    elif map_var == "sdcc_binary_constraints":
        assert cc_metric is None
        direction = kwargs["direction"]
        cc_metric = round(coverage_results["results"][t]["SDCC"], 3)
        target_name = direction.split("-")[0]
        source_name = direction.split("-")[1]

        title = f"Set Difference Coverage of {t}-way Interactions of {target_name} not in {source_name} in {dataset_name}, | (SDCC = {cc_metric})"
        file_basename = (
            f"sdcc_binary_t{t}-{dataset_name}_{target_name}_not_in_{source_name}"
        )

    return title, file_basename


def map_info_var(map_var, kwargs):
    if map_var == "binary":
        vmin = 0
        vmax = 1
        bw = [
            sns.color_palette(palette="Greys", as_cmap=False)[1],
            sns.color_palette(palette="Greys", as_cmap=False)[5],
        ]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("bw", bw, len(bw))
        cbar_kws = {
            "label": "Binary Coverage in Dataset",
            "ticks": [0, 1],
            "ticklabels": ["Not Covered", "Covered"],
        }

    elif map_var == "frequency":
        vmin = 0
        vmax = None
        cmap = sns.cubehelix_palette(as_cmap=True, hue=3)
        cbar_kws = {
            "label": "Number of Appearences per Interaction",
            "ticks": [0, 10000],
            "ticklabels": ["Less Frequent", "More Frequent"],
        }

    elif map_var == "proportion_frequency":
        vmin = 0
        vmax = 1
        cmap = sns.cubehelix_palette(as_cmap=True, start=2.8, rot=0.1)
        cbar_kws = {
            "label": textwrap.fill(
                "Proportion of Appearences per Interaction for Combination", 55
            ),
            "ticks": [0, 1],
            "ticklabels": ["Lower Proportion", "Higher Proportion"],
        }

    elif map_var == "proportion_frequency_standardized":
        vmin = None
        vmax = None
        cmap = sns.diverging_palette(145, 300, s=60, as_cmap=True)
        cbar_kws = {
            "label": textwrap.fill(
                "Proportion of Appearences per Interaction for Combination, standardized",
                55,  # + r"$\frac{n_{jl}-\frac{N}{c_j}}{N}$"
            ),
            "ticks": [0, 1],
            "ticklabels": ["Underrepresentation", "Overrepresentation"],
        }

    elif map_var == "performance":
        vmin = 0
        vmax = 1
        cmap = sns.cubehelix_palette(as_cmap=True, rot=-0.4)
        cbar_kws = {
            "label": f"Performance of Samples belonging to Interaction, {kwargs['metric']}",
            "ticks": [0, 1],
            "ticklabels": ["Low (0)", "High (1)"],
        }

    elif map_var == "sdcc_binary_constraints":
        vmin = -1
        vmax = 1
        cmap = sns.color_palette("rocket", as_cmap=True, n_colors=3)
        # cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        #   "rocket_colors", rocket_colors, len(rocket_colors))
        cbar_kws = {
            "label": "Set Difference Constraints",
            "ticks": [-1, 0, 1],
            "ticklabels": ["Not in Set", "In Intersection", "In Difference"],
        }

    elif map_var == "sdcc_binary_constraints_wneither":
        vmin = -1
        vmax = 2
        cmap = sns.color_palette("rocket", as_cmap=True, n_colors=4)
        # cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        #   "rocket_colors", rocket_colors, len(rocket_colors))
        cbar_kws = {
            "label": "Set Difference Constraints",
            "ticks": [-1, 0, 1, 2],
            "ticklabels": [
                "Not in Set",
                "In Intersection",
                "In Difference",
                "In Neither Set",
            ],
        }
    else:
        raise NameError

    cbar = None
    return vmin, vmax, cmap, cbar_kws


def set_colorbar(cbar: matplotlib.colorbar.Colorbar, cbar_kws: dict):
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(cbar_kws["label"], fontsize=AXIS_SIZE)

    cbar.set_ticks(cbar_kws["ticks"])
    cbar.set_ticklabels(cbar_kws["ticklabels"], rotation=-30)

    return cbar
