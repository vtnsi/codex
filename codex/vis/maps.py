import seaborn as sns
import numpy as np
import textwrap
import matplotlib.colors

from ..utils import codex_metrics as metrics
import matplotlib.pyplot as plt

TITLE_SIZE = 18
AXIS_SIZE = int(0.8*TITLE_SIZE)

def set_plot_axes(title):
    plt.gcf()

    plt.title(title, weight="bold", fontsize=18, pad=15)
    plt.xlabel("Interactions", fontsize=AXIS_SIZE, labelpad=15, weight='bold')
    plt.ylabel("Combinations", fontsize=AXIS_SIZE, labelpad=15, weight='bold')

    
    return

def save_plots():
    #savefig
    return

def map_info_data():

    return

def map_info_txtl():

    return


def map_info_var():

    return

def __set_colorbar(colorbar, cbar_kws, mode):
    colorbar.ax.get_yaxis().labelpad = 15
    colorbar.ax.tick_params(labelsize=12)
    colorbar.set_label(cbar_kws['label'], fontsize=AXIS_SIZE)

    if mode == "sdcc_binary_constraints_neither":
        colorbar.set_ticks([-0.667, 0, 0.667, 1.667])
        colorbar.set_ticklabels(cbar_ticklabels, rotation=-30)
    elif mode == "sdcc_binary_constraints":
        colorbar.set_ticks([-0.667, 0, 0.667])
        colorbar.set_ticklabels(cbar_ticklabels, rotation=-30)
    elif mode == "binary":
        colorbar.set_ticks([0, 1])
        colorbar.set_ticklabels(cbar_ticklabels, rotation=-30)

    return

def frequency_map(square, counts):
    """
    Fill square map valued on an interaction's number of appearances.
    """
    # Takes in a list of lists that represents interaction coverage
    # produces a map and colors the times an interaction appears
    # 0 = not covered, x > 0 --> covered x times, -1 indicates invalid interaction
    # one list per rank, but rank lists have different size
    # one row per list, possibly with white space. Raw counts.
    for row in range(0, len(counts)):
        for column in range(0, len(counts[row])):
            if counts[row][column] > 0:
                square[row][column] = counts[row][column]
            else:
                assert square[row][column] == -1
                # square[row][column] = 0

    cmap = sns.cubehelix_palette(as_cmap=True, hue=3)
    cbar_kws = {"label": "Number of Appearences per Interaction"}
    cbar_ticklabels = ["Less Frequent", "More Frequent"]

    return square, cmap, cbar_kws, cbar_ticklabels


def frequency_map_function(square, counts, funct=None, funct_name=None):
    """
    Fill square map valued on a function applied to an interaction's
    number of appearances.
    """
    # Takes in a list of lists that represents interaction coverage
    # produces a map and colors the times an interaction appears
    # 0 = not covered, x > 0 --> covered x times, -1 indicates invalid interaction
    # one list per rank, but rank lists have different size
    # one row per list, possibly with white space. A specified function
    # is applied to raw counts to map to some other interval if needed

    for row in range(0, len(counts)):
        for column in range(0, len(counts[row])):
            if counts[row][column] > 0:
                try:
                    square[row][column] = funct(counts[row][column])
                except:
                    raise NameError("No function applicable.")
            else:
                assert square[row][column] == -1
                # square[row][column] = 0

    cmap = sns.cubehelix_palette(as_cmap=True, hue=4)
    cbar_kws = {
        "label": textwrap.fill(
            "{}(# of Appearences per Interaction)".format(funct_name), 55
        )
    }
    cbar_ticklabels = ["Less Frequent", "More Frequent"]


    return square, cmap, cbar_kws


def frequency_map_proportion(square, counts):
    """
    Fill square map valued on the proportion an interaction makes for
    a combination's total number of interactions.
    """
    for row in range(0, len(counts)):
        for column in range(0, len(counts[row])):
            if counts[row][column] > 0:
                square[row][column] = counts[row][column] / np.sum(counts[row])
            else:
                # assert square[row][column] == -1
                square[row][column] = 0
    cmap = sns.cubehelix_palette(as_cmap=True, start=2.8, rot=0.1)
    # cmap = sns.diverging_palette(145, 300, s=60, as_cmap=True)
    cbar_kws = {
        "label": textwrap.fill(
            "Proportion of Appearences per Interaction for Combination", 55
        )
    }
    cbar_ticklabels = ["Lower Proportion", "Higher Proportion"]


    # print(square)
    return square, cmap, cbar_kws, cbar_ticklabels


def frequency_map_proportion_standardized(square, counts):
    """
    Fill square map valued on the proportion an interaction makes for
    a combination's total number of interactions.
    """
    for row in range(0, len(counts)):
        for column in range(0, len(counts[row])):
            if counts[row][column] > 0:
                square[row][column] = metrics.standardized_proportion_per_interaction(
                    counts, row, column
                )
                # square[row][column] = counts[row][column]/np.sum(counts[row])
            else:
                # assert square[row][column] == -1
                square[row][column] = -999
    # cmap = sns.cubehelix_palette(as_cmap=True, start=2.8, rot=0.1)
    cmap = sns.diverging_palette(145, 300, s=60, as_cmap=True)
    cbar_kws = {
        "label": textwrap.fill(
            "Proportion of Appearences per Interaction for Combination, "
            + "Standardized proportion",
            55,
            #", "
            #+ r"$\frac{n_{jl}-\frac{N}{c_j}}{N}$"
        )
    }
    cbar_ticklabels = ["Underrepresented", "Overrepresented"]

    # print(square)
    return square, cmap, cbar_kws, cbar_ticklabels


def performance_map(square, counts, perf, metric):
    """
    Fill square map valued on a model's performance on samples of each t-way interaction
    appearing in the dataset.
    """
    for row in range(0, len(counts)):
        for column in range(0, len(counts[row])):
            if perf[row][column] is not None:
                square[row][column] = perf[row][column]
            else:
                square[row][column] = -1
                # assert square[row][column] == -999

            """if counts[row][column] > 0:
                square[row][column] = perf[metric][row][column]
            else:
                assert square[row][column] == -1"""

    cmap = sns.cubehelix_palette(as_cmap=True, rot=-0.4)
    cbar_kws = {
        "label": textwrap.fill(
            "Performance of Samples of Interaction, {}".format(metric), 55
        )
    }
    cbar_ticklabels = ["Low", "High"]

    return square, cmap, cbar_kws, cbar_ticklabels


def frequency_map_binary(square, counts):
    """
    Fill square map valued on its appearance/absence.
    """
    for row in range(0, len(counts)):
        for column in range(0, len(counts[row])):
            if counts[row][column] > 0:
                square[row][column] = 1
            else:
                square[row][column] = 0
                # assert square[row][column] == -1

    # cmap = sns.cubehelix_palette(as_cmap=True, hue=0)
    # print(sns.color_palette(palette='Greys', as_cmap=False)[2])
    bw = [
        sns.color_palette(palette="Greys", as_cmap=False)[1],
        sns.color_palette(palette="Greys", as_cmap=False)[5],
    ]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("bw", bw, len(bw))
    cbar_kws = {"label": "Binary Coverage in Dataset"}
    cbar_tick_labels = ["Not Covered", "Covered"]

    return square, cmap, cbar_kws, cbar_tick_labels


# Takes in a list of lists that represents interaction coverage
# produces a map and colors the times an interaction appears
# 0 = not covered, x > 0 --> covered x times, -1 indicates invalid interaction
# one list per rank, but rank lists have different size
# one row per list, possibly with white space

# Takes in a list of lists that represents interaction coverage
# produces a map and colors the times an interaction appears
# 0 = not covered, x > 0 --> covered x times, -1 indicates invalid interaction (in constraint not in T)
# one list per rank, but rank lists have different size
# one row per list, possibly with white space


def sd_map_binary_constrained(square, counts):
    """
    SDCC map
    """
    # cmap = sns.cubehelix_palette(as_cmap=True, hue=15)
    # cmap = sns.color_palette('plasma', as_cmap=True)
    rocket_colors = sns.color_palette("rocket", as_cmap=False, n_colors=3)  # [:3]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "rocket_colors", rocket_colors, len(rocket_colors)
    )

    cmap.set_over("k")
    cmap.set_under("w")
    cbar_kws = {"label": "Set Difference Constraints", "ticks": [-1, 0, 1]}

    cbar_ticklabels = ["Not in Set", "In Intersection", "In Difference"]

    for row in range(0, len(counts)):
        for column in range(0, len(counts[row])):
            square[row][column] = counts[row][column]

    return square, cmap, cbar_kws, cbar_ticklabels


def sd_map_binary_constrained_neither(square, counts, source_name=None):
    """
    SDCC map
    """
    # cmap = sns.cubehelix_palette(as_cmap=True, hue=15)
    # cmap = sns.color_palette('plasma', as_cmap=True)
    rocket_colors = sns.color_palette("rocket", as_cmap=False, n_colors=4)  # [:4]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "rocket_colors", rocket_colors, len(rocket_colors)
    )
    cmap.set_over("k")
    cmap.set_under("w")
    cbar_kws = {"label": "Set Difference Constraints", "ticks": [-1, 0, 1, 2]}

    cbar_ticklabels = [
        "Not in {} Set".format(source_name),
        "In Intersection",
        "In Set Difference",
        "In Neither Set",
    ]

    for row in range(0, len(counts)):
        for column in range(0, len(counts[row])):
            square[row][column] = counts[row][column]

    return square, cmap, cbar_kws, cbar_ticklabels
