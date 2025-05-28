import numpy as np
import copy


def stock_results_empty(codex_input, dataset_name, model_name, universe, **kwargs):
    coverage_results = {
        "info": {"dataset name": dataset_name, "model name": model_name},
        "universe": universe,
        "mode": codex_input["mode"],
    }

    for kwarg in kwargs:
        coverage_results[kwarg] = kwargs[kwarg]

    return coverage_results


def get_counts_performance(interactions_probe, interactions_exploit, t):
    bottom_common = [
        interaction
        for interaction in interactions_probe[t]["bottom interactions"]
        if interaction in interactions_exploit[t]["bottom interactions"]
    ]
    bottom_probe_perf = [
        interactions_probe[t][interaction]["performance"]
        for interaction in bottom_common
    ]
    bottom_exploit_perf = [
        interactions_exploit[t][interaction]["performance"]
        for interaction in bottom_common
    ]

    order = np.argsort(bottom_probe_perf)
    bottom_common = [bottom_common[i] for i in order]
    bottom_probe_perf = [bottom_probe_perf[i] for i in order]
    bottom_exploit_perf = [bottom_exploit_perf[i] for i in order]

    return bottom_common, bottom_probe_perf, bottom_exploit_perf


def combine_counts_and_performance(coverage_dict, t, metric):
    interaction_performance = coverage_dict[t]["human readable performance"][metric]
    interaction_counts = coverage_dict[t]["combination counts"]

    interaction_consolidated = {}
    for i, rank in enumerate(interaction_performance):
        for j, interaction in enumerate(interaction_performance[rank]):
            if interaction_performance[rank][interaction] == None:
                continue
            interaction_consolidated[interaction] = {
                "performance": None,
                "counts": None,
            }
            interaction_consolidated[interaction]["performance"] = (
                interaction_performance[rank][interaction]
            )
            interaction_consolidated[interaction]["counts"] = interaction_counts[i][j]

    return interaction_consolidated


def sort_interactions(interaction_dict, t, display_num, order=None):
    interactions = list(interaction_dict[t].keys())
    performance = [
        interaction_dict[t][interaction]["performance"] for interaction in interactions
    ]
    counts = [
        interaction_dict[t][interaction]["counts"] for interaction in interactions
    ]

    sort_by_performance = np.argsort(performance)
    interactions_sorted = [interactions[i] for i in sort_by_performance]
    performance_sorted = [performance[i] for i in sort_by_performance]
    counts_sorted = [counts[i] for i in sort_by_performance]
    return interactions_sorted


def consolidated_interaction_info(coverage, strengths, metric, order, display_n):
    # coverage = kwargs['coverage']
    # display_num = kwargs['display_num']
    # metric = kwargs['metric']

    interactions = {}
    for t in strengths:
        # t = str(t)
        interactions[t] = combine_counts_and_performance(coverage, t, metric)

        bottom_interactions = sort_interactions(interactions, t, display_n, order)
        top_interactions = copy.deepcopy(bottom_interactions)
        top_interactions.reverse()

        if "ascending" in order:
            interactions[t]["bottom interactions"] = bottom_interactions
        if "descending" in order:
            interactions[t]["top interactions"] = top_interactions

    return interactions
