import os
import pandas as pd
import numpy as np
import textwrap
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from cycler import cycler
import matplotlib
import warnings
from matplotlib.pyplot import cm

title_size = 18
titlepad = 16
labelpad = 12
axis_size = int(4 * title_size / 5)

ALPHA = 0.55
MODEL_NAME_MAP = {
    "gnb": {
        "decoded": ["Gaussian Naive Bayes"],
        "encoded": ["gnb"],
        "color": "tab:blue",
    },
    "lr": {
        "decoded": ["Logistic Regression"],
        "encoded": ["lr"],
        "color": "tab:orange",
    },
    "rf": {"decoded": ["Random Forest"], "encoded": ["rf"], "color": "tab:green"},
    "knn": {"decoded": ["K-Nearest Neighbors"], "encoded": ["knn"], "color": "tab:red"},
    "svm": {"decoded": ["SVM"], "encoded": ["svm"], "color": "tab:purple"},
    "all": {
        "name": ["All Models"],
        "decoded": [
            "Gaussian Naive Bayes",
            "Logistic Regression",
            "Random Forest",
            "KNN",
            "SVM",
        ],
        "encoded": ["gnb", "lr", "rf", "knn", "svm"],
    },
}


def overall_and_pi_plots(
    output_dir,
    coverage_by_skew_single_combo: dict,
    combo_of_interest,
    interaction_of_interest,
    metrics,
    chosen_model=None,
    t=2,
):
    combination_interaction_perf_xskew(
        output_dir,
        coverage_by_skew_single_combo,
        combo_of_interest,
        interaction_of_interest,
        chosen_model,
        metrics=["accuracy"],
        t=t,
    )  # ACCURACY HARDCODE
    overall_perf_xskew(
        output_dir,
        coverage_by_skew_single_combo,
        combo_of_interest,
        interaction_of_interest,
        chosen_model,
        metrics=metrics,
    )
    plt.cla()


# Performance of interactions
def combination_interaction_perf_xskew(
    output_dir,
    coverage_by_skew_single_combo: dict,
    combo_of_interest: str,
    interaction_of_interest: str,
    chosen_model,
    metrics,
    t=2,
):
    assert type(metrics) is list
    #################
    # cm = plt.get_cmap('Set3')

    # you can also set line style, width

    ####################

    for metric in metrics:
        try:
            human_readable = coverage_by_skew_single_combo["low"][chosen_model][
                "coverage"
            ][t]["human readable performance"][metric]
        except:
            t = str(t)
            human_readable = coverage_by_skew_single_combo["low"][chosen_model][
                "coverage"
            ][t]["human readable performance"][metric]

        interaction_perf_std = []
        for combination_diffsort in human_readable[metric]:
            for interaction in human_readable[metric][combination_diffsort]:
                interaction_across_skew = [
                    coverage_by_skew_single_combo[skew][chosen_model][t][
                        "human readable performance"
                    ][metric][interaction]
                    for skew in coverage_by_skew_single_combo
                    if skew != "output_dir" and skew != "baseline"
                ]
                interaction_perf_std.append(np.std(interaction_across_skew))
        print(interaction_perf_std)

        plt.cla()
        for combination in tqdm(human_readable):
            sample_interaction_name = list(human_readable[combination].keys())[0]
            combination_pieces = sample_interaction_name.split("'")
            combination_decoded = "({}*{})".format(
                combination_pieces[1], combination_pieces[5]
            )

            interaction_names_sorted_alphabetical = sorted(
                list(human_readable[combination].keys())
            )
            human_readable_sorted = {
                k: v
                for k, v in sorted(
                    human_readable[combination].items(), key=lambda item: item[1]
                )
            }
            interaction_names_sorted = list(human_readable_sorted.keys())
            print(interaction_names_sorted)

            """matplotlib.rcParams["axes.prop_cycle"] = cycler(
                color=[cm(v) for v in np.linspace(0, 1, len(interaction_names_sorted))]
            )"""

            plt.figure(figsize=(12, 8))
            limit_plots_10 = True
            num_plots = 0
            for i, interaction in enumerate(interaction_names_sorted):
                print("PLOTTING {}".format(interaction))
                interaction_perf_list_by_skew = [
                    coverage_by_skew_single_combo[skew][chosen_model]["coverage"][t][
                        "human readable performance"
                    ][metric][combination][interaction]
                    for skew in coverage_by_skew_single_combo
                    if skew != "output_dir" and skew != "baseline"
                ]

                plt.plot(interaction_perf_list_by_skew, label=interaction, marker="o")
                """
                plt.axhline(y=coverage_by_skew['baseline'][model_code]['coverage']['info']['Overall Performance'][metric], 
                    linestyle='dashed', 
                    label='baseline',
                    alpha=0.3)
                """
                model_name = MODEL_NAME_MAP[chosen_model]["decoded"][0]
                combo_of_interest_split = (
                    combo_of_interest.strip("(").strip(")").split("*")
                )
                interaction_of_interest_split = (
                    interaction_of_interest.strip("(").strip(")").split("*")
                )

                title_txt = textwrap.fill(
                    "{} Per-Interaction Performance of Interactions of Combination {} on Datasets of Varying Bias in Combination ({}={}), ({}={})".format(
                        model_name,
                        combination_decoded,
                        combo_of_interest_split[0],
                        interaction_of_interest_split[0],
                        combo_of_interest_split[1],
                        interaction_of_interest_split[1],
                    ),
                    60,
                )
                ylab_txt = "Per-Interaction Model Performance: {}".format(metric)
                xlab_txt = "Levels of bias"
                plt.title(title_txt, pad=titlepad, weight="bold", fontsize=title_size)
                plt.grid(visible=True, axis="y")
                plt.xlabel(xlab_txt, fontsize=axis_size, labelpad=labelpad)
                plt.ylabel(ylab_txt, fontsize=axis_size, labelpad=labelpad)
                plt.ylim((0, 1))
                # plt.yscale('log')
                plt.xticks(
                    ticks=[0, 1, 2],
                    labels=[
                        skew
                        for skew in coverage_by_skew_single_combo
                        if skew != "output_dir" and skew != "baseline"
                    ],
                )

                plt.legend()
                plt.tight_layout()

                if limit_plots_10:
                    if (i % 10 == 0 and i != 0) or (
                        i == len(human_readable[combination]) - 1
                    ):
                        print(
                            "SAVING TO {}".format(
                                os.path.join(
                                    output_dir,
                                    "perf_vs_skew-to_{}-displaying_{}-{}-{}-num_{}.png".format(
                                        combo_of_interest,
                                        combination_decoded,
                                        chosen_model,
                                        metric,
                                        num_plots,
                                    ),
                                )
                            )
                        )
                        plt.savefig(
                            os.path.join(
                                output_dir,
                                "perf_vs_skew-to_{}-displaying_{}-{}-{}-num_{}.png".format(
                                    combo_of_interest,
                                    combination_decoded,
                                    chosen_model,
                                    metric,
                                    num_plots,
                                ),
                            )
                        )
                        plt.clf()
                        num_plots += 1

                else:
                    if i == len(human_readable[combination]) - 1:
                        print(
                            "SAVING TO {}".format(
                                os.path.join(
                                    output_dir,
                                    "perf_vs_skew-to_{}-displaying_{}-{}-{}-num_{}.png".format(
                                        combo_of_interest,
                                        combination_decoded,
                                        chosen_model,
                                        metric,
                                        num_plots,
                                    ),
                                )
                            )
                        )
                        plt.savefig(
                            os.path.join(
                                output_dir,
                                "perf_vs_skew-to_{}-displaying_{}-{}-{}-num_{}.png".format(
                                    combo_of_interest,
                                    combination_decoded,
                                    chosen_model,
                                    metric,
                                    num_plots,
                                ),
                            )
                        )
                        num_plots += 1
                        plt.clf()

        plt.cla()


def overall_perf_xskew(
    output_dir,
    coverage_by_skew: dict,
    combo_of_interest: str,
    interaction_of_interest: str,
    chosen_model,
    metrics,
):
    """ovr_perf_list = [coverage_by_skew[skew][chosen_model]['coverage']['info']['Overall Performance'][metric] for skew in coverage_by_skew
    if skew != 'output_dir' and skew != 'baseline]"""

    for metric in metrics:
        model_list = MODEL_NAME_MAP[chosen_model]["encoded"]
        model_name_list = MODEL_NAME_MAP[chosen_model]["decoded"]

        plt.figure(figsize=(12, 8))

        for i, model_code in enumerate(model_list):
            print(model_code)
            model_name = model_name_list[i]

            ovr_perf_list = [
                coverage_by_skew[skew][model_code]["coverage"]["info"][
                    "Overall Performance"
                ][metric]
                for skew in coverage_by_skew
                if skew != "output_dir" and skew != "baseline"
            ]

            # plt.figure(figsize=(12,8))
            if len(model_list) == 1:
                plt.plot(
                    ovr_perf_list, color="red", marker="o", label=model_name_list[i]
                )
                plt.axhline(
                    y=coverage_by_skew["baseline"][model_code]["coverage"]["info"][
                        "Overall Performance"
                    ][metric],
                    linestyle="dashed",
                    alpha=ALPHA,
                    color="red",
                )
            else:
                model_name = MODEL_NAME_MAP[chosen_model]["name"][0]
                plt.plot(
                    ovr_perf_list,
                    marker="o",
                    label=model_name_list[i],
                    color=MODEL_NAME_MAP[model_code]["color"],
                )
                plt.axhline(
                    y=coverage_by_skew["baseline"][model_code]["coverage"]["info"][
                        "Overall Performance"
                    ][metric],
                    linestyle="dashed",
                    alpha=ALPHA,
                    color=MODEL_NAME_MAP[model_code]["color"],
                )

        combo_of_interest_split = combo_of_interest.strip("(").strip(")").split("*")
        interaction_of_interest_split = (
            interaction_of_interest.strip("(").strip(")").split("*")
        )
        title_txt = textwrap.fill(
            "{} Overall Model Performance on Datasets Biased on: ({}={}), ({}={})".format(
                model_name,
                combo_of_interest_split[0],
                interaction_of_interest_split[0],
                combo_of_interest_split[1],
                interaction_of_interest_split[1],
            ),
            60,
        )

        plt.xticks(
            ticks=[0, 1, 2],
            labels=[
                skew
                for skew in coverage_by_skew
                if skew != "output_dir" and skew != "baseline"
            ],
        )
        plt.xlabel("Levels of bias", fontsize=axis_size, labelpad=labelpad)
        plt.title(title_txt, pad=titlepad, weight="bold", fontsize=title_size)
        plt.ylim((0, 1))
        plt.grid(visible=True, axis="y")
        plt.ylabel(
            "Overall model performance: {}".format(metric),
            fontsize=axis_size,
            labelpad=labelpad,
        )
        plt.legend()
        plt.savefig(
            os.path.join(
                output_dir,
                "overall_performance_by_skew-{}-{}-{}.png".format(
                    combo_of_interest, chosen_model, metric
                ),
            )
        )

        plt.cla()


def overall_perf_xskew_xmetrics(
    output_dir,
    coverage_by_skew: dict,
    combo_of_interest: str,
    interaction_of_interest: str,
    chosen_model,
    metrics,
):
    """ovr_perf_list = [coverage_by_skew[skew][chosen_model]['coverage']['info']['Overall Performance'][metric] for skew in coverage_by_skew
    if skew != 'output_dir']"""

    model_list = MODEL_NAME_MAP[chosen_model]["encoded"]
    model_name_list = MODEL_NAME_MAP[chosen_model]["decoded"]

    print(__name__, chosen_model, model_list, model_name_list)

    plt.figure(figsize=(12, 8))

    for i, model_code in enumerate(model_list):
        print(model_code)
        model_name = model_name_list[i]

        for metric in metrics:
            ovr_perf_list = [
                coverage_by_skew[skew][model_code]["coverage"]["info"][
                    "Overall Performance"
                ][metric]
                for skew in coverage_by_skew
                if skew != "output_dir" and skew != "baseline"
            ]

            model_name = MODEL_NAME_MAP[chosen_model]["decoded"][0]
            plt.plot(ovr_perf_list, marker="o", label=metric)

        plt.axhline(
            y=coverage_by_skew["baseline"][model_code]["coverage"]["info"][
                "Overall Performance"
            ][metric],
            linestyle="dashed",
            color="magenta",
            label="baseline",
        )

        combo_of_interest_split = combo_of_interest.strip("(").strip(")").split("*")
        interaction_of_interest_split = (
            interaction_of_interest.strip("(").strip(")").split("*")
        )
        title_txt = textwrap.fill(
            "{} Overall Model Performance on Datasets Biased on: ({}={}), ({}={}), All Metrics".format(
                model_name,
                combo_of_interest_split[0],
                interaction_of_interest_split[0],
                combo_of_interest_split[1],
                interaction_of_interest_split[1],
            ),
            60,
        )

        plt.xticks(
            ticks=[0, 1, 2],
            labels=[
                skew
                for skew in coverage_by_skew
                if skew != "output_dir" and skew != "baseline"
            ],
        )
        plt.xlabel("Levels of bias", fontsize=axis_size, labelpad=labelpad)
        plt.title(title_txt, pad=titlepad, weight="bold", fontsize=title_size)
        plt.ylim((0, 1))
        plt.grid(visible=True, axis="y")
        plt.ylabel(
            "Overall model performance: {}".format(metrics),
            fontsize=axis_size,
            labelpad=labelpad,
        )
        plt.legend()
        plt.savefig(
            os.path.join(
                output_dir,
                "overall_performance_by_skew-{}-{}-{}.png".format(
                    combo_of_interest, chosen_model, "all_metrics"
                ),
            )
        )

        plt.cla()


def construct_table(
    output_dir,
    coverage_by_skew: dict,
    combo_of_interest: str,
    interaction_of_interest: str,
    chosen_model,
    metrics,
):
    per_combo_df = pd.DataFrame()
    global MODEL_NAME_MAP
    for metric in metrics:
        model_list = MODEL_NAME_MAP["all"]["encoded"]
        model_name_list = MODEL_NAME_MAP["all"]["decoded"]

        single_metric_df = pd.DataFrame(
            columns=["model-metric"]
            + [
                skew
                for skew in coverage_by_skew[combo_of_interest]
                if skew != "output_dir"
            ]
        )
        training_sizes = {
            skew: coverage_by_skew[combo_of_interest][skew]["training_size"]
            for skew in coverage_by_skew[combo_of_interest]
            if skew != "output_dir"
        }
        training_sizes["model-metric"] = "training set size"
        single_metric_df.loc[len(single_metric_df)] = training_sizes

        for i, model_code in enumerate(model_list):
            perf_per_skew = {
                skew: coverage_by_skew[combo_of_interest][skew][model_code]["coverage"][
                    "info"
                ]["Overall Performance"][metric]
                for skew in coverage_by_skew[combo_of_interest]
                if skew != "output_dir"
            }
            perf_per_skew["model-metric"] = "{}-{}".format(
                MODEL_NAME_MAP[model_code]["decoded"][0], metric
            )
            single_metric_df.loc[len(single_metric_df)] = perf_per_skew

        per_combo_df = pd.concat((per_combo_df, single_metric_df))

    per_combo_df.to_csv(
        os.path.join(output_dir, "meta_analysis-{}.csv".format(combo_of_interest))
    )

    return per_combo_df


def main():
    return
