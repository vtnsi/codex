# Author Erin Lanus lanus@vt.edu, leebri2n@vt.edu
# Created Date: Mar 1 2023
# Updated Date: 23 September, 2024

if __name__ == "__main__":
    MODE_CLI = True
else:
    MODE_CLI = False

import sys
import os
import logging
import shutil
import glob

import json
import pandas as pd
from tqdm import tqdm

import directory_tree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from codex.modules import combinatorial, output, binning
from codex.utils import input_handler as input_handler, universe_handler, results_handler, dataset_handler, prereq_handler as prereq_check
sie_ml = None; sie_analysis=None


def setup_new_codex_env(dir_name=None, parent_dir="", templates=False, tutorial=False):
    existing_codex_dirs = glob.glob(os.path.realpath(os.path.join(parent_dir, dir_name+'*')))

    # exec_dir = os.path.dirname(os.path.realpath(__file__))
    exec_dir = "../"
    try:
        assert os.path.exists(os.path.join(exec_dir, "resources", "templates"))
    except:
        exec_dir = './'
    

    if dir_name is None or dir_name == "":
        dir_name = "new_codex_dir"

    if len(existing_codex_dirs)  == 0:
        codex_dir_new = os.path.realpath(os.path.join(parent_dir, f'{dir_name}'))
    else:
        codex_dir_new = os.path.realpath(os.path.join(parent_dir, f'{dir_name}_{len(existing_codex_dirs)}'))
    os.makedirs(codex_dir_new, exist_ok=False)

    for component in ["binning", "configs", "splits", "performance", "datasets", "runs", "universe"]:
        subdir = os.path.join(codex_dir_new, component)
        os.makedirs(subdir, exist_ok=True)

    if templates:
        for filename in os.listdir(os.path.join(exec_dir, "resources", "templates")):
            for subdir in os.listdir(codex_dir_new):
                if str(subdir)[:-1] in str(filename):
                    shutil.copy(os.path.join(exec_dir, "resources", "templates", filename), os.path.join(codex_dir_new, subdir, filename))
    if tutorial:
        for filename in os.listdir(os.path.join(exec_dir, "resources", "tutorial")):
            for subdir in os.listdir(codex_dir_new):
                if str(subdir)[:-1] in str(filename):
                    shutil.copy(os.path.join(exec_dir, "resources", "tutorial", filename), os.path.join(codex_dir_new, subdir, filename))

    directory_tree.DisplayTree(codex_dir_new)
    return codex_dir_new


def run(codex_input, verbose: str='1'):
    """
    Allows for different modes that can lead to different functionalities
    modes:
    - dataset evaluation -- run the cc for a single dataset
    - dataset split  -- split a dataset into train validation test and run cc and sdcc on the parts, assume no split file
    - dataset split evaluation -- run cc and sdcc on the parts given by the split file
    - transfer learning -- assumes two separate dataset files with the features specified in input json in common and runs sdcc on them
    - transfer learning dataset augmentation -- also performs image selection from target to augment source
    - performance by interaction -- identifies every t-strength interaction of features in a dataset and relates model performance of samples containing an interaction.
    """
    codex_input = input_handler.handle_input_file(codex_input)
    output_dir, strengths = input_handler.define_experiment_variables(codex_input)
    timed = codex_input["timestamp"]
    
    logger_level, filename = output.logger_parameters(verbose, output_dir=output_dir, timed=timed)
    output.intialize_logger(__name__, logger_level, filename)
    output.intialize_logger(output.__name__, logger_level, filename)
    output.intialize_logger(binning.__name__, logger_level, filename)
    output.intialize_logger(combinatorial.__name__, logger_level, filename)

    mode = codex_input["mode"]

    if mode == "dataset evaluation":
        result = dataset_evaluation(codex_input)

    elif mode == "dataset split evaluation":
        source_name, target_name = prereq_check.dse_prereq_check(codex_input)
        split, performance, metric = input_handler.extract_sp(codex_input)
        result = dataset_split_evaluation(
            codex_input, split, source_name, target_name, comparison=False
        )

    elif mode == "dataset split comparison":
        split_consolidated, performance_consolidated, metric = input_handler.extract_sp(
            codex_input
        )
        result = dataset_split_comparison(
            codex_input, split_consolidated, performance_consolidated, metric
        )

    elif mode == "performance by interaction":
        split, performance, metric = input_handler.extract_sp(codex_input)
        result = performance_by_interaction(
            codex_input, split, performance, metric, coverage_subset="train"
        )

    elif mode == "model probe":
        split_p, split_e, perf_p, perf_e, metric = (
            input_handler.extract_sp_partitioning(codex_input)
        )
        result = model_probe(codex_input, split_p, split_e, perf_p, perf_e, metric)

    elif mode == "balanced test set construction":
        test_set_size_goal = codex_input["test_set_size_goal"]
        result = balanced_test_set_construction(codex_input, test_set_size_goal)

    elif mode == "sie":
        test_set_size_goal = codex_input["test set size goal"]
        result = systematic_inclusion_exclusion(codex_input, test_set_size_goal)
    elif mode == "sie rfml":
        test_set_size_goal = codex_input["test set size goal"]
        result = systematic_inclusion_exclusion_iq(codex_input)
    elif mode == "sie analysis":
        test_set_size_goal = codex_input["test set size goal"]
        result = systematic_inclusion_exclusion_binomial_linreg(
            codex_input, "aggregate_SIE_performance-rareplanes-linreg.csv"
        )
    elif mode == "sie demo":
        test_set_size_goal = codex_input["test set size goal"]
        result = systematic_inclusion_exclusion_demo(codex_input, test_set_size_goal)

    elif mode == "biasing":
        skew_levels = codex_input["Skew levels"]
        result = performance_by_frequency_coverage(codex_input, skew_levels)

    else:
        raise NameError("Mode not found!")

    print("Results saved to {}.".format(output_dir))
    return result


# --- Entry point functions ----#


def dataset_evaluation(codex_input):
    """
    Dataset evaluation comptues combinatorial coverage on a dataset.

    Parameters:
    codex_input: dict
        JSON config dictionary.
    """
    dataset_name, model_name = input_handler.extract_names(codex_input)
    output_dir, strengths = input_handler.define_experiment_variables(codex_input)
    universe, dataset_df = universe_handler.define_input_space(codex_input)

    coverage_results = results_handler.stock_results_empty(
        codex_input, dataset_name, model_name, universe
    )
    for t in strengths:
        coverage_results[t] = combinatorial.CC_main(
            dataset_df, dataset_name, universe, t, output_dir
        )

    coverage_results_formatted = output.dataset_eval_vis(
        output_dir, dataset_name, coverage_results, strengths
    )
    return coverage_results_formatted


# dataset split evaluation
def dataset_split_evaluation(
    codex_input: dict,
    split: dict,
    source_name="train",
    target_name="test",
    comparison=False,
):
    """
    Dataset split evaluation computes SDCC from a split and plots it against its resultant
    model performance. Split file required to contain at least train and test splits.

    Parameters
    input: dict
        Read from the input file containing experiment requirements, pathing, info.

    split: dict
        Read from a JSON file whose keys are the split partition and values are lists of
        the sample ID's as presented in the dataset.

    performance: dict
        Read from a JSON file containing a model's performance on a test set.

    metric: str
        Chosen metric to evaluate performance for the experiment.

    sdcc_direction: str
        Direction of the set difference between the target and source datasets.
    """
    # produces a dataframe for each of the train/val/test subsets given the ids provided in the split file
    dataset_name, model_name = input_handler.extract_names(codex_input)
    output_dir, strengths = input_handler.define_experiment_variables(codex_input)
    universe, dataset_df = universe_handler.define_input_space(codex_input)

    split_id = split["split_id"]
    sample_id_col = codex_input["sample_id_column"]
    source_ids = split[source_name]
    target_ids = split[target_name]
    traindf = dataset_handler.df_slice_by_id_reorder(
        sample_id_col, dataset_df, sample_ids=source_ids
    )
    testdf = dataset_handler.df_slice_by_id_reorder(
        sample_id_col, dataset_df, sample_ids=target_ids
    )
    if "validation" in split:
        val_ids = split["validation"]
        val_df = dataset_handler.df_slice_by_id_reorder(
            sample_id_col, dataset_df, sample_ids=val_ids
        )

    coverage_results = results_handler.stock_results_empty(
        codex_input, dataset_name, model_name, universe, split_id=split_id
    )
    for t in strengths:
        coverage_results[t] = combinatorial.SDCC_main(
            traindf,
            source_name,
            testdf,
            target_name,
            universe,
            t,
            output_dir,
            comparison_mode=comparison,
            split_id=split_id,
        )
        if "validation" in split:
            coverage_results[t].update(
                combinatorial.SDCC_main(
                    traindf,
                    source_name,
                    val_df,
                    "val",
                    universe,
                    t,
                    output_dir,
                    comparison_mode=comparison,
                    split_id=split_id,
                )
            )

    coverage_results_formatted = output.dataset_split_eval_vis(
        output_dir,
        dataset_name,
        coverage_results,
        strengths,
        split_id,
        comparison=comparison,
    )
    return coverage_results_formatted


def dataset_split_comparison(
    codex_input: dict, split_multiple: dict, performance_multiple: dict, metric: str
):
    """
    Dataset split comparison compares SDCC values from multiple splits
    with their resultant model performances agaisnt eachother by calling
    dataset split evaluation for each split.

    Split files required to contain at least train and test splits.

    Parameters
    input: JSON
        Read from the input file containing experiment requirements, pathing, info.

    split_list: list
        A list of file paths pointing to JSON files whose keys are the split partition
        and values are lists of the sample ID's as presented in the dataset.

    performance_list: list
        A list of JSON files containing model performance expressed as a
        named metric. References the split file used to obtain model
        performance. Only overall model performance is required for this mode.

    metric: str
        Chosen metric to evaluate performance for the experiment.

    sdcc_direction: str
        Direction of the set difference between the target and source datasets.
    """
    dataset_name, model_name = input_handler.extract_names(codex_input)
    output_dir, strengths = input_handler.define_experiment_variables(codex_input)
    universe, dataset_df = universe_handler.define_input_space(codex_input)

    split_ids = [split_multiple[split]["split_id"] for split in split_multiple]
    coverage_results = results_handler.stock_results_empty(
        codex_input, dataset_name, model_name, universe, split_id=split_ids
    )
    for split_file in split_multiple:
        split_id = split_multiple[split_file]["split_id"]
        coverage_results[split_id] = dataset_split_evaluation(
            input, split_multiple[split_file], comparison=True
        )
    if performance_multiple is None:
        return coverage_results

    coverage_results = output.dataset_split_comp_vis(
        output_dir,
        dataset_name,
        coverage_results,
        performance_multiple,
        strengths,
        metric,
        split_ids=split_ids,
    )
    return coverage_results


def performance_by_interaction(
    codex_input,
    split,
    performance,
    metric,
    coverage_subset="train",
    display_n=10,
    order="ascending descending",
    probing=False,
):
    """
    Codex mode that studies performance of each interaction in the dataset by aggregating
    per-sample performance of samples containing a particular interaction.

    Parameters:
    input: JSON
        Read from the input file containing experiment requirements, pathing, info.
    split: dict
        Read from a JSON file whose keys are the split partition and values are lists of
        the sample ID's as presented in the dataset.
    performance: dict
        Read from a JSON file containing per-sample model performance on a test set.
    metric: str
        Chosen metric to evaluate performance for the experiment.
    display_n: int
        Number of interactions to compare in each of the test set.
    order: str
        Display the n lowest interactions, or n highest interactions as appearing in the test set.

    Returns:
    results: dict
        Coverage results and human readable performance of the model on the test sets.

    Raises:
    KeyError
        If IDs of split file do not match those of per-sample performance section.
    """
    dataset_name, model_name = input_handler.extract_names(codex_input)
    output_dir, strengths = input_handler.define_experiment_variables(codex_input)
    universe, dataset_df = universe_handler.define_input_space(codex_input)
    sample_id_col = codex_input["sample_id_column"]

    # SORT, THEN RESET INDEX AND DROP
    train_ids = split["train"]
    test_ids = split["test"]
    train_df_sorted = dataset_handler.reorder_df_by_sample(
        sample_id_col,
        dataset_handler.df_slice_by_id_reorder(sample_id_col, dataset_df, train_ids),
    )
    test_df_sorted = dataset_handler.reorder_df_by_sample(
        sample_id_col,
        dataset_handler.df_slice_by_id_reorder(sample_id_col, dataset_df, test_ids),
    )
    performance_df = dataset_handler.reorder_df_byindex(
        performance["test"]["Per-Sample Performance"]
    )

    coverage_results = results_handler.stock_results_empty(
        codex_input, dataset_name, model_name, universe
    )
    # coverage_results_sdcc = results_handler.stock_results_empty(codex_input, dataset_name, model_name, universe)
    for t in strengths:
        coverage_results[t] = combinatorial.performanceByInteraction_main(
            test_df_sorted,
            train_df_sorted,
            performance_df,
            dataset_name,
            universe,
            t,
            output_dir,
            metric,
            sample_id=sample_id_col,
            coverage_subset=coverage_subset,
        )
        # coverage_results_sdcc[t] = combinatorial.SDCC_main(trainDF, 'train', testDF, 'test', universe, t, output_dir, split_id=split_id)
    try:
        coverage_results['info']['Overall Performance'] = performance['test']['Overall Performance']
    except:
        print("No overall performance section found...")
        
    coverage_results = output.performance_by_interaction_vis(
        output_dir,
        dataset_name,
        coverage_results,
        strengths,
        metric,
        order,
        display_n,
        subset=coverage_subset,
    )  # , display_all_lr=display_all_ranks)
    
    
    return coverage_results


def performance_by_interaction_partitioning(
    input, probe_name="included", exploit_name="excluded", force_write=False
):
    """
    Codex mode that studies performance of each interaction appearing in subsets of a
    dataset by aggregating per-sample performance of samples containing a particular
    interaction appearing in each set, named the probing and exploit set, separating them.

    Parameters:
    - input: JSON
        Read from the input file containing experiment requirements, pathing, info.
    - probe_name: str
        Key designating the name of the probing subset as it appers in the split and per-sample
        performance files.
    - exploit_name: str
        Key designating the name of the exploit subset as it appaers in the split and per-sample
        performance files.
    - force_write: Bool
        Overwrites probe and exploit coverage results even if either one does not exist in the
        output directory.

    Returns:
    - probe_coverage, exploit_coverage: JSON
        Newly calculated coverage results and per-sample human readable performance on both
        the probe and exploit test subsets.

    Raises:
    - KeyError: probe or exploit split ID's or per-sample performance is not found under the
        subset names provided.
    """
    output_dir, strengths = input_handler.define_experiment_variables(input)

    probe_output_dir = os.path.join(output_dir, probe_name)
    exploit_output_dir = os.path.join(output_dir, exploit_name)
    probe_coverage_file = os.path.join(probe_output_dir, "coverage.json")
    exploit_coverage_file = os.path.join(exploit_output_dir, "coverage.json")
    if not os.path.exists(probe_output_dir):
        os.makedirs(probe_output_dir)
    if not os.path.exists(exploit_output_dir):
        os.makedirs(exploit_output_dir)

    if (
        not os.path.exists(probe_coverage_file)
        or not os.path.exists(exploit_coverage_file)
        or force_write
    ):
        # CODEX SUB-OP
        performance_by_interaction(input, subset="_{}".format(probe_name))
        performance_by_interaction(input, subset="_{}".format(exploit_name))
    with open(probe_coverage_file) as f:
        probe_coverage = json.load(f)
    with open(exploit_coverage_file) as f:
        exploit_coverage = json.load(f)

    return probe_coverage, exploit_coverage


def model_probe(
    input,
    split_p,
    split_e,
    perf_p,
    perf_e,
    metric,
    name_p="_included",
    name_e="_excluded",
    display_n=10,
    order="ascending descending",
):
    """
    Model probing computes performance by interaction on two partitions of a test set - a probing set
    and exploit set - to compare how high-performing and low-performing interactions of the probing set
    perform in a potential exploit set and highlight the model's vulnerabilities.

    Parameters
    - input: JSON
        Read from the input file containing experiment requirements, pathing, info.
    - split_p: dict
        Read from a JSON file whose keys are the split partition and values are lists of
        the sample ID's as presented in the dataset, with the probing partition.
    - performance_p: dict
        Read from a JSON file containing overall model performance on the probing partition
        of the test set as well as per-sample performance performed by the model on each sample
        in the probe set.
    - split_e: dict
        Read from a JSON file whose keys are the split partition and values are lists of
        the sample ID's as presented in the dataset, with the exploit partition.
    - performance_e: dict
        Read from a JSON file containing overall model performance on the exploit partition
        of the test set as well as per-sample performance performed by the model on each sample
        in the exploit set.
    - metric: str
        Chosen metric to evaluate performance for the experiment.
    display_n: int
        Number of interactions to compare in each of the probe and exploit sets.
    order: str
        Display the n lowest interactions, or n highest interactions as appearing in the probe set.

    Returns:
    - results: dict
        Coverage results and human readable performance of probing and exploit sets.
    """
    output_dir, strengths = input_handler.define_experiment_variables(input)
    probe_coverage = performance_by_interaction(
        input, split_p, perf_p, metric, subset=name_p, withhold=True
    )
    exploit_coverage = performance_by_interaction(
        input, split_e, perf_e, metric, subset=name_e, withhold=True
    )

    result = output.model_probing_vis(
        output_dir,
        name_p,
        name_e,
        probe_coverage,
        exploit_coverage,
        strengths,
        metric,
        order,
        display_n,
        subset="train",
    )
    return result


# construct a balanced test set


def balanced_test_set_construction(
    input, test_set_size_goal, include_baseline=True, shuffle=False, adjusted_size=None, form_exclusions=False
):
    """
    Runs test set post optimization in order to construct as balanced a test set as possible
    balancing appearance of various interactions. (???)

    Parameters
    - input: dict
        Read from the input file containing experiment requirements, pathing, info.
    - test_set_size_goal: int
        The desired number of samples for each balanced test set.

    Returns:
    - result: dict
        Coverage results on the overall dataset as well as split designations for each withheld
        interaction resulting from post-test set optimzation
    """
    output_dir, strengths = input_handler.define_experiment_variables(input)
    universe, dataset_df = universe_handler.define_input_space(input)

    if shuffle:
        dataset_df = dataset_df.sample(len(dataset_df))

    logging.getLogger("universe").info(universe)

    if adjusted_size is None:
        result = combinatorial.balanced_test_set(
            dataset_df,
            input["dataset_name"],
            input["sample_id_column"],
            universe,
            strengths,
            int(test_set_size_goal),
            output_dir,
            include_baseline=include_baseline,
            form_exclusions=form_exclusions
        )
    else:
        result = combinatorial.balanced_test_set(
            dataset_df,
            input["dataset_name"],
            input["sample_id_column"],
            universe,
            strengths,
            int(adjusted_size),
            output_dir,
            include_baseline=include_baseline,
            form_exclusions=form_exclusions
        )

    if not os.path.exists(os.path.join(output_dir, "splits_by_json")):
        os.makedirs(os.path.join(output_dir, "splits_by_json"))
    if not os.path.exists(os.path.join(output_dir, "splits_by_csv")):
        os.makedirs(os.path.join(output_dir, "splits_by_csv"))

    for k in result.keys():
        if "model_" in k:
            output.output_json_readable(
                result[k],
                write_json=True,
                file_path=os.path.join(output_dir, "splits_by_json", k + ".json"),
            )
    result["universe"] = universe

    output.output_json_readable(
        result, write_json=True, file_path=os.path.join(output_dir, "coverage.json")
    )

    return result


def systematic_inclusion_exclusion_demo(codex_input, test_set_size_goal):
    output_dir, strengths = input_handler.define_experiment_variables(codex_input)
    filename = "aggregate_SIE_performance-rareplanes-linreg.csv"
    for t in strengths:
        print(
            "Generating splits excluding each {}-way interactions for combinations {}...".format(
                t, codex_input["features"]
            )
        )
        result = balanced_test_set_construction(codex_input, test_set_size_goal)

        SIE_splits = {
            key.split("model")[-1]: result[key]
            for key in result.keys()
            if "model" in key
        }
        SIE_ids = list(SIE_splits.keys())
        print(
            "Generated {} splits excluding interactions for combinations {}...".format(
                len(SIE_ids), codex_input["features"]
            )
        )

        print("BINOMIAL REGRESSION TESTING FOR SIGNIFICANCE, INCLUDED IN TRAINING:")
        systematic_inclusion_exclusion_binomial_linreg(codex_input, filename)

    return result


def systematic_inclusion_exclusion(
    codex_input, test_set_size_goal, training_params=None
):
    output_dir, strengths = input_handler.define_experiment_variables(codex_input)

    data_dir = codex_input["original_data_directory"]
    dataset_dir_YOLO = codex_input["dataset_directory"]
    training_dir = codex_input["training_directory"]
    if dataset_dir_YOLO == "":
        if not os.path.exists(os.path.join(output_dir, "datasets")):
            os.makedirs(os.path.join(output_dir, "datasets"))
        dataset_dir_YOLO = os.path.join(output_dir, "datasets")

    result = None

    if os.path.exists(os.path.join(output_dir, "coverage.json")):
        overwrite_splits = (
            input(
                "Balanced test set splits already found for {}. Overwrite splits? (y/n): ".format(
                    output_dir
                )
            )
            == "y"
        )
        if overwrite_splits:
            shutil.rmtree(output_dir)
            result = balanced_test_set_construction(codex_input, test_set_size_goal)
    else:
        result = balanced_test_set_construction(codex_input)

    with open(os.path.join(output_dir, "coverage.json")) as f:
        result = json.load(f)

    n_t = result["universal_test_set_size"]
    n_g = codex_input["test set size goal"]
    proceed_wsize = (
        input(
            "Assembled a test set of {} samples compared to goal of {}. Proceed? (y/n): ".format(
                n_t, n_g
            )
        )
        == "y"
    )
    if not proceed_wsize:
        adjusted_size = input("New test set goal (int): ")
        result = balanced_test_set_construction(
            codex_input, adjusted_size=adjusted_size
        )

    with open(os.path.join(output_dir, "coverage.json")) as f:
        result = json.load(f)

    if result is None:
        with open(os.path.join(output_dir, "coverage.json")) as f:
            result = json.load(f)
    SIE_splits = {
        key.split("model")[-1]: result[key] for key in result.keys() if "model" in key
    }
    SIE_ids = list(SIE_splits.keys())
    print(SIE_ids)

    # CONSIDER ASSIGNING DATASET ACCORDING TO EACH CONFIG DIRECTORY
    train = False
    if train:
        pass
        sie_ml.data_distribution_SIE(
            SIE_splits, data_dir, dataset_dir_YOLO, overwrite=True, mode_text=True
        )
        sie_ml.train_SIE(
            SIE_splits,
            dataset_dir_YOLO,
            training_dir,
            epochs=300,
            batch=128,
            devices=[0, 1, 2, 3],
            force_resume=True,
        )

    score = True
    if score:
        sie_ml.evaluate(
            SIE_splits, data_dir, dataset_dir_YOLO, training_dir, config_dir=output_dir
        )
    table_filename = ""
    analyze = True

    if analyze:
        systematic_inclusion_exclusion_binomial_linreg(codex_input, table_filename)

    return SIE_splits


def systematic_inclusion_exclusion_iq(codex_input, test_set_size_goal):
    iq_data_dir = "../../rfml-datagen/dataset-0912-0/"
    config_path = "/home/hume-users/leebri2n/PROJECTS/dote_1070-1083/py-waspgen/configs/mod_classifier.json"
    output_dir, strengths = input_handler.define_experiment_variables(codex_input)
    write_single_files = False
    
    # Awaiting PyWASPgen public release
    sie_iq = None

    if write_single_files:
        df, mod_classes = sie_iq.gen_dataset(
            iq_data_dir,
            1000000,
            0,
            0,
            config_path,
            overwrite=False,
            metadata_save_dir=os.path.join(
                codex_input["codex_directory"], codex_input["data_directory"]
            ),
            write_single_files=write_single_files,
        )
    else:
        iq_streams, burst_list, df, mod_classes = sie_iq.gen_dataset(
            iq_data_dir,
            10000,
            0,
            0,
            config_path,
            overwrite=True,
            metadata_save_dir=os.path.join(
                codex_input["codex_directory"], codex_input["data_directory"]
            ),
            write_single_files=write_single_files,
        )

    train_params = {
        "epochs": 15,
        "num_devices": 1,
        "num_workers": 16,
        "batch_size": 4096,
    }
    training_dir = "../../codex-use_cases/rfml/training"

    result = balanced_test_set_construction(codex_input, test_set_size_goal)
    sie_splits = {
        key.split("model")[-1]: result[key] for key in result.keys() if "model" in key
    }

    perf = {"model{}".format(id): None for id in sie_splits}
    for i, id in enumerate(tqdm(sie_splits)):
        print("SIE SPLIT: {}".format(id))
        model = None
        if i == 3:
            break

        train_data = None
        val_data = None
        if not write_single_files:
            train_data, val_data, test_incl_data, test_excl_data = (
                sie_iq.handle_streams(
                    iq_data_dir, sie_splits[id], id, iq_streams, burst_list
                )
            )

        model = sie_iq.iq_train_SIE(
            train_params,
            sie_splits[id],
            id,
            data_dir=iq_data_dir,
            mod_classes=mod_classes,
            training_dir=training_dir,
            train_data_precomputed=train_data,
            val_data_precomputed=val_data,
        )  # iq_data_dir, mod_classes, train_params=train_params, sie_split=sie_splits[id])
        perf["model{}".format(id)] = sie_iq.iq_eval_SIE(
            model,
            iq_data_dir,
            mod_classes,
            batch_size=16,
            sie_split=sie_splits[id],
            sie_id=id,
        )
        output.output_json_readable(
            result,
            print_json=False,
            write_json=True,
            file_path=os.path.join(output_dir, "coverage{}.json".format(id)),
        )

    result.update(perf)
    output.output_json_readable(
        result,
        print_json=False,
        write_json=True,
        file_path=os.path.join(output_dir, "coverage_sie.json"),
    )
    return result


def systematic_inclusion_exclusion_binomial_linreg(codex_input, table_filename):
    perf_table = pd.read_csv(
        os.path.join(
            codex_input["codex_directory"],
            codex_input["performance_folder"],
            table_filename,
        )
    )
    metric = codex_input["metric"]
    metrics = codex_input["metrics"]
    features = codex_input["features"]

    output_dir, strengths = input_handler.define_experiment_variables(codex_input)

    results = {}
    if metric == "all":
        for metric in metrics:
            model_summary, contrasts_summary, contrast_names = (
                sie_analysis.SIE_binomial_regression_main(
                    perf_table, metrics, metric, features
                )
            )
            results = results.update(
                output.SIE_regression_test_vis(
                    output_dir, model_summary, contrasts_summary, contrast_names
                )
            )
    else:
        model_summary, contrasts_summary, contrast_names = (
            sie_analysis.SIE_binomial_regression_main(
                perf_table, metrics, metric, features
            )
        )
        results = output.SIE_regression_test_vis(
            output_dir, model_summary, contrasts_summary, contrast_names
        )

    return results

def performance_by_frequency_coverage(codex_input, skew_levels:list, test_set_size_goal=250):
    import utils.pbfc_biasing as biasing

    output_dir, strengths = input_handler.define_experiment_variables(input)
    universe, dataset_df_init = universe_handler.define_input_space(input)
    
    result = balanced_test_set_construction(codex_input, test_set_size_goal, form_exclusions=False)
    
    for t in strengths:
        results_all_models = combinatorial.performance_by_frequency_coverage_main()

    output.output_json_readable(results_all_models, write_json=True, 
                                file_path=os.path.join(output_dir, 'pbcf.json'))

    return results_all_models

def main(kwargs):
    try:
        setup_new_dir = (str.lower(kwargs['setup_new_dir']) == 'true')
    except KeyError:
        setup_new_dir = None
    
    if setup_new_dir is not None:
        try:  
            new_dir_name = kwargs['name']
        except:
            raise KeyError("In creating a new CODEX directory; requires <name of CODEX directory>.")
        try:
            new_dir_parent = kwargs['parent_dir']
        except:
            new_dir_parent = '..'
            raise KeyError("In creating a new CODEX directory; requires <parent directory of CODEX directory>.")

        try:
            include_templates = (str.lower(kwargs['include_templates']) == 'true')
        except KeyError:
            include_templates = False
        try:
            include_tutorial = (str.lower(kwargs['include_examples']) == 'true')
        except KeyError:
            include_tutorial = False
        setup_new_codex_env(new_dir_name, new_dir_parent, templates=include_templates, tutorial=include_tutorial)   
        return

    input_fp = kwargs["input"]
    try:
        verbosity = str(kwargs["verbose"])

        if verbosity not in ['1', '2']:
            raise NameError("Output verbosity levels supported by CODEX: 1 - coarse; 2 - fine.")
    except KeyError:
        verbosity = str(1)

    with open(input_fp) as f:
        codex_input = json.load(f)

    output_dir, strengths = input_handler.define_experiment_variables(codex_input)
    run(codex_input, verbosity)
    return

"""
Use keyword args on command line to pass important info
    $ python combinatorial.py path=./ sourceName=RarePlanes sourceFile=RarePlanesMetadata_process_binned.csv mode=cc t=2 exp=MsA
    $ python codex.py input=input.json
"""
if __name__ == "__main__":
    if len(sys.argv) < 1:
        raise KeyError("Improper command line. Input file required. For input file named input.json, format is python codex.py input=input.json verbose=[1/2]")
        
    kwargs = dict(arg.split("=") for arg in sys.argv[1:])

    main(kwargs)
