# Author: Erin Lanus lanus@vt.edu
# Updated Date: 4 April 2024
# Movement to new repo: 12 Apr, 2024

# Refactoring: 06 June, 2025

# -------------------- Imported Modules --------------------
import os
import math
import numpy as np
import pandas as pd
import json
import copy
import logging
from scipy.special import comb

import combinatorial.abstraction as abstraction
from combinatorial.abstraction import Mapping
from output import output
from modes import pbfc_biasing as biasing, pbfc_data

# -------------------- Global Variables               --------------------#
label_centric = False
abstraction.label_centric = label_centric

verbose = False
identifyImages = False

LOGGER_COMBI = logging.getLogger(__name__)


# -------------------- Functions Compute Coverage     -------------------- #


# main entry point for computing CC on one file
# expects name for labeling on plots
def CC_main(dataset_df, dataset_name, universe, strengths, output_dir):
    """
    Main entry point to computing combinatorial coverage over data.
    """

    global verbose, label_centric, identifyImages
    label_centric = False
    mMap = Mapping(universe["features"], universe["levels"], None)
    data = abstraction.encoding(dataset_df, mMap, True)

    LOGGER_COMBI.log(msg="Metadata level map:\n{}".format(mMap), level=15)
    LOGGER_COMBI.log(msg="Data representation:\n{}".format(data), level=15)

    k = len(data.features)

    cc_results = {t: {} for t in strengths}
    for t in strengths:
        if t > k:
            print("t =", t, " cannot be greater than number of features k =", k)
            # return
            continue

        # computes CC for one dataset
        CC = combinatorialCoverage(data, t)

        LOGGER_COMBI.log(msg="CC:{}".format(CC), level=15)

        counts = CC["countsAllCombinations"]
        ranks = decodeCombinations(data, CC, t)

        decodedMissing = abstraction.decode_missing_interactions(
            data, computeMissingInteractions(data, CC)
        )
        # create t file with results for this t -- CC and missing interactions list
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(os.path.join(output_dir, "CC")):
            os.makedirs(os.path.join(output_dir, "CC"))

        ranks = decodeCombinations(data, CC, t)
        output.writeCCtToFile(output_dir, dataset_name, t, CC)
        output.writeMissingtoFile(output_dir, dataset_name, t, decodedMissing)

        cc_results[t] = cc_dict(CC)
        cc_results[t]["combinations"] = ranks
        cc_results[t]["combination counts"] = counts
        cc_results[t]["missing interactions"] = decodedMissing

    return cc_results


# main entry point for computing SDCC on two files
# expects sourceName for labeling on plots and sourceFile as file name of binned csv file
# and same for target


# (path, sourceName, sourceFile, targetName, targetFile, metadata, t):
def SDCC_main(
    sourceDF,
    sourceName,
    targetDF,
    targetName,
    dataset_name,
    universe,
    strengths,
    output_dir,
    comparison_mode,
    split_id,
):
    """
    Main entry point to computing set difference combinatorial coverage between two
    datasets. Set difference meaning, that which is appearing in a target dataset
    that does not exist in a source dataset.
    """
    global verbose, label_centric, identifyImages

    mMap = Mapping(universe["features"], universe["levels"], None)
    source_data = abstraction.encoding(sourceDF, mMap, True)
    target = abstraction.encoding(targetDF, mMap, True)

    LOGGER_COMBI.log(msg="Metadata level map:\n{}".format(mMap), level=15)
    LOGGER_COMBI.log(
        msg="'Source' data representation:\n{}".format(source_data), level=15
    )
    LOGGER_COMBI.log(msg="'Target' data representation:\n{}".format(target), level=15)

    k = len(source_data.features)

    setdif_name = f"{targetName}-{sourceName}"
    setdif_name_reversed = f"{sourceName}-{targetName}"

    sdcc_results = {
        sourceName: {"results": {}},
        targetName: {"results": {}},
        setdif_name: {"results": {}},
        setdif_name_reversed: {"results": {}},
    }

    for t in strengths:
        if t > k:
            print("t =", t, " cannot be greater than number of features k =", k)
            # return
            continue

        sourceCC = combinatorialCoverage(source_data, t)
        targetCC = combinatorialCoverage(target, t)

        LOGGER_COMBI.log(msg="Source CC:\n".format(sourceCC), level=15)
        LOGGER_COMBI.log(msg="Target CC:\n".format(targetCC), level=15)

        if comparison_mode:
            output_dir = output.create_output_dir(os.path.join(output_dir, split_id))
        else:
            output_dir = output.create_output_dir(output_dir)

        source_ranks = decodeCombinations(source_data, sourceCC, t)
        target_ranks = decodeCombinations(target, targetCC, t)
        assert source_ranks == target_ranks

        sourceDecodedMissing = abstraction.decode_missing_interactions(
            source_data, computeMissingInteractions(source_data, sourceCC)
        )
        targetDecodedMissing = abstraction.decode_missing_interactions(
            target, computeMissingInteractions(target, targetCC)
        )

        # compute set difference target \ source
        SDCCconstraints = setDifferenceCombinatorialCoverageConstraints(
            sourceCC, targetCC
        )
        output.writeSDCCtToFile(output_dir, sourceName, targetName, t, SDCCconstraints)
        setDifferenceInteractions = computeSetDifferenceInteractions(
            target, SDCCconstraints
        )
        decodedSetDifferenceInteractions = (
            abstraction.decode_set_difference_interactions(
                target, setDifferenceInteractions
            )
        )
        output.writeSetDifferencetoFile(
            output_dir, sourceName, targetName, t, decodedSetDifferenceInteractions
        )

        if identifyImages and len(decodedSetDifferenceInteractions) > 0:
            IDS = identifyImagesWithSetDifferenceInteractions(
                targetDF, decodedSetDifferenceInteractions
            )
            output.writeImagestoFile(output_dir, sourceName, targetName, t, IDS)
            print(
                "number of target images containing an interaction not present in source: ",
                len(IDS),
            )

        # compute opposite direction source \ target
        reverseSDCCconstraints = setDifferenceCombinatorialCoverageConstraints(
            targetCC, sourceCC
        )
        output.writeSDCCtToFile(
            output_dir, targetName, sourceName, t, reverseSDCCconstraints
        )
        reversesetDifferenceInteractions = computeSetDifferenceInteractions(
            source_data, reverseSDCCconstraints
        )
        reversedecodedSetDifferenceInteractions = (
            abstraction.decode_set_difference_interactions(
                source_data, reversesetDifferenceInteractions
            )
        )
        output.writeSetDifferencetoFile(
            output_dir,
            targetName,
            sourceName,
            t,
            reversedecodedSetDifferenceInteractions,
        )

        # if identifyImages and len(decodedSetDifferenceInteractions) > 0:
        #            identifyImagesWithSetDifferenceInteractions(sourceDF, reversedecodedSetDifferenceInteractions)

        # See commit before e736442
        sdcc_results[sourceName]["results"][t] = cc_dict(sourceCC)
        sdcc_results[sourceName]["results"][t]["combinations"] = source_ranks
        sdcc_results[sourceName]["results"][t]["combination counts"] = sourceCC[
            "countsAllCombinations"
        ]
        sdcc_results[sourceName]["results"][t]["missing interactions"] = (
            sourceDecodedMissing
        )

        sdcc_results[targetName]["results"][t] = cc_dict(targetCC)
        sdcc_results[targetName]["results"][t]["missing interactions"] = (
            targetDecodedMissing
        )
        sdcc_results[targetName]["results"][t]["combinations"] = target_ranks
        sdcc_results[targetName]["results"][t]["combination counts"] = targetCC[
            "countsAllCombinations"
        ]
        sdcc_results[targetName]["results"][t]["missing interactions"] = (
            targetDecodedMissing
        )

        sdcc_results[setdif_name]["results"][t] = sdcc_dict(SDCCconstraints)
        sdcc_results[setdif_name]["results"][t]["combinations"] = source_ranks
        sdcc_results[setdif_name]["results"][t]["sdcc counts"] = SDCCconstraints[
            "setDifferenceInteractionsCounts"
        ]

        sdcc_results[setdif_name_reversed]["results"][t] = sdcc_dict(
            reverseSDCCconstraints
        )
        sdcc_results[setdif_name_reversed]["results"][t]["combinations"] = source_ranks
        sdcc_results[setdif_name_reversed]["results"][t]["sdcc counts"] = (
            reverseSDCCconstraints["setDifferenceInteractionsCounts"]
        )

    return sdcc_results


def performanceByInteraction_main(
    train_dataDF: pd.DataFrame,
    test_dataDF: pd.DataFrame,
    performance_df: pd.DataFrame,
    dataset_name: str,
    universe: dict,
    strengths: list,
    output_dir: str,
    metrics: list,
    coverage_subset=None,
):
    """
    Main entry point for computing performance by interaction on a test set and combinatorial
    coverage over a training set from data.
    """
    global verbose, label_centric, identifyImages
    label_centric = False
    mMap = Mapping(universe["features"], universe["levels"], None)

    data_test = abstraction.encoding(test_dataDF, mMap, True)
    data_train = abstraction.encoding(train_dataDF, mMap, True)

    LOGGER_COMBI.log(msg="Metadata level map:\n{}".format(mMap), level=15)
    LOGGER_COMBI.log(msg="Data representation:\n{}".format(data_test), level=15)

    if performance_df.index.values.tolist() != test_dataDF.index.values.tolist():
        raise KeyError(
            "IDs in performance file do not match IDs in test set of split file"
        )

    k = len(data_test.features)
    pbi_results = {t: {} for t in strengths}
    for t in strengths:
        if t > k:
            raise ValueError(
                "t = {} cannot be greater than number of features k = {}".format(t, k)
            )

        # computes CC for one dataset as well as the performance
        CC_test, perf = computePerformanceByInteraction(
            data_test, t, performance_df=performance_df
        )
        CC_train = combinatorialCoverage(data_train, t)

        if coverage_subset == "train":
            CC = CC_train
        elif coverage_subset == "test":
            CC = CC_test
        else:
            raise KeyError(
                "Coverage over subset {} not found in split file.".format(
                    coverage_subset
                )
            )

        LOGGER_COMBI.log(level=15, msg="CC over train: {}".format(CC))

        decodedMissing = abstraction.decode_missing_interactions(
            data_test, computeMissingInteractions(data_test, CC)
        )
        # create t file with results for this t -- CC and missing interactions list

        output.writeCCtToFile(output_dir, dataset_name, t, CC)
        output.writeMissingtoFile(output_dir, dataset_name, t, decodedMissing)

        train_ranks = decodeCombinations(data_train, CC, t)
        test_ranks = decodeCombinations(data_test, CC, t)
        assert test_ranks == train_ranks

        pbi_results[t] = cc_dict(CC)
        pbi_results[t]["combinations"] = train_ranks
        pbi_results[t]["combination counts"] = CC["countsAllCombinations"]

        pbi_results[t]["missing interactions"] = decodedMissing

        pbi_results["performance"] = perf
        pbi_results["human readable performance"] = (
            abstraction.decode_performance_grouped_combination(data_test, CC, perf)
        )

    return pbi_results


def performance_by_frequency_coverage_main(
    trainpool_df,
    test_df_balanced,
    entire_df_cont,
    universe,
    t,
    output_dir,
    skew_level,
    id=None,
):
    EXP_NAME = id

    combination_list = biasing.get_combinations(universe, t)
    for combination in combination_list:
        indices_per_interaction, combination_names = biasing.interaction_indices_t2(
            df=trainpool_df
        )

        (
            train_df_biased,
            train_df_selected_filename,
            combo_int_selected,
            interaction_selected,
        ) = biasing.skew_dataset_relative(
            df=trainpool_df,
            interaction_indices=indices_per_interaction,
            skew_level=skew_level,
            extract_combination=combination,
            output_dir=output_dir,
        )

        # TO EDIT (052825): ~~~~~~~~~~~~
        test_df_STATIC_1211 = pd.DataFrame()
        original_data_filename = ""
        drop_list = []
        classifier = None
        split_dir = ""
        performance_dir = ""
        INPUT_DICT = None
        scaler = None
        metric = None
        results_multiple_model = None
        jsondict = None
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        train_df = entire_df_cont.loc[
            entire_df_cont.index.isin(train_df_biased.index.tolist())
        ]
        test_df = entire_df_cont.loc[
            entire_df_cont.index.isin(test_df_STATIC_1211.index.tolist())
        ]

        full_df_combo = pd.concat([train_df, test_df], axis=0)
        full_df_cont_combo_filename = "{}-{}_Skewed.csv".format(
            original_data_filename, EXP_NAME
        )
        # full_df_combo.to_csv(os.path.join(data_dir_skew, full_df_cont_combo_filename))

        X_train, X_test, Y_train, Y_test, split_filename = pbfc_data.prep_split_data(
            None,
            train_df=train_df,
            test_df=test_df,
            name=EXP_NAME,
            drop_list=drop_list,
            split_dir=split_dir,
            target_col="Diabetes_binary",
            id_col="ID",
        )

        perf_filenames = classifier.model_suite(
            X_train,
            Y_train,
            X_test,
            Y_test,
            experiment_name=EXP_NAME,
            output_dir=performance_dir,
            scaler=scaler,
        )

        for perf_filename in perf_filenames:
            input_dict_new = copy.deepcopy(INPUT_DICT)

            if "_gnb" in perf_filename:
                model_name = "Gaussian Naive Bayes"
                model_name_small = "gnb"
            elif "_lr" in perf_filename:
                model_name = "Logistic Regression"
                model_name_small = "lr"
            elif "_rf" in perf_filename:
                model_name = "Random Forest"
                model_name_small = "rf"
            elif "_knn" in perf_filename:
                model_name = "KNN"
                model_name_small = "knn"
            elif "_svm" in perf_filename:
                model_name = "SVM"
                model_name_small = "svm"
            else:
                print("No model name found!")

            # Static
            input_dict_new["metric"] = metric
            input_dict_new["dataset_file"] = full_df_cont_combo_filename
            input_dict_new["split_file"] = split_filename
            # Change per model
            save_dir = "_runs/pbi_pipeline/pbi-{}-{}".format(EXP_NAME, model_name_small)
            input_dict_new["config_id"] = save_dir
            input_dict_new["model_name"] = model_name
            input_dict_new["dataset_name"] = "CDC Diabetes, skewed {}, {}".format(
                skew_level, model_name
            )
            input_dict_new["performance_file"] = perf_filename

            # CODEX ~~~~~~~~~~~~~~~
            # of chosen combo, skew_level, model
            result = None  # codex.run(input_dict_new, verbose="1")
            results_multiple_model[model_name_small] = {
                "coverage": result,
                "save_dir": save_dir,
            }

    print("CHECK DOESNT CHANGE:", interaction_selected)
    results_multiple_model["interaction_skewed"] = interaction_selected
    results_multiple_model["training_size"] = len(train_df_biased)

    return jsondict


# computes the combinatorial coverage metric of a given representation for a given t
def combinatorialCoverage(representation, t):
    """
    Computes the combinatorial coverage metric of a given representation for a given t.

    Parameters:
    representation: Mapping
        Encoded representation of a dataset

    t: int
        Combination strength of the t-way interactions examined

    Returns:
    coverageDataStructure: dict
        Data structure containing coverage info for each t, including the universe,
        appearing interactions, their counts
    """

    totalPossibleInteractions = 0
    countAppearingInteractions = 0
    countsAllCombinations = []
    k = len(representation.features)  # number of columns in input file
    coverageDataStructure = {
        "t": t,
        "k": k,
        "representation": representation,
        "countsAllCombinations": countsAllCombinations,
    }
    kprime = k
    tprime = t
    if label_centric:  # if label_centric,
        tprime -= 1  # require t-1 way interactions of all other columns
        kprime -= 1  # and only consider the first k-1 columns in the rank
    # k choose t - number of  combinations of t columns
    kct = comb(kprime, tprime, exact=True)

    LOGGER_COMBI.log(
        level=25,
        msg="k: {}\n k': {}\n, t: {}\n, t': {}\n, k choose t: {}".format(
            k, kprime, t, tprime, kct
        ),
    )

    coverageDataStructure = {
        "t": t,
        "k": k,
        "kct": kct,
        "representation": representation,
        "countsAllCombinations": countsAllCombinations,
    }

    for rank in range(0, kct):  # go rank by rank
        # enumerate all combinations
        set = []  # start with an empty set to pass through to the function
        # find the set of columns corresponding to this rank
        set = abstraction.rank_to_tuple(set, rank, tprime)
        interactionsForRank = 1  # compute v1*v2*...*vt so start with 1
        if label_centric:  # if every combination must include the label column
            set.append(kprime)  # add the last column (label) to the set
        # it's t not t' as every combination is of t columns regardless
        for column in range(0, t):
            interactionsForRank *= len(representation.values[set[column]])
        totalPossibleInteractions += interactionsForRank  # update the total
        # countsEachCombination COULD be boolean BUT we want to get all counts for later diversity metrics
        # create the empty list for counting
        countsEachCombination = [0 for index in range(0, interactionsForRank)]
        # count the combinations that appear in this rank by checking all of the rows in the columns indicated by set[]
        for row in range(0, len(representation.data)):
            index = abstraction.convert_value_combo_to_index(representation, set, row)
            symbols = []

            # NOTE, 06.06.24: this loop is huge
            # symbols = abstraction.convert_index_to_value_combo(representation, index, set, t)
            # logging.getLogger(__name__).debug('Row \t{}\t has symbols: {} corresponding to index {}'.format(row, symbols, index))
            countsEachCombination[index] += 1

        countsAllCombinations.append(countsEachCombination)
        LOGGER_COMBI.log(
            level=15,
            msg="Rank:{}, set:{}, combinations:{}".format(
                rank, set, interactionsForRank
            ),
        )
        LOGGER_COMBI.log(
            level=15, msg="Counts of each combination:{}".format(countsEachCombination)
        )

        # update the count -- since might be more than 0, count if a cell is nonzero rather than summing the counts
        for index in range(0, interactionsForRank):
            if countsEachCombination[index] > 0:
                countAppearingInteractions += 1
    coverageDataStructure["totalPossibleInteractions"] = totalPossibleInteractions
    coverageDataStructure["countAppearingInteractions"] = countAppearingInteractions
    return coverageDataStructure


# computes the set difference of T \ S given the combinatorial coverage representation of both has already been computed
# binary decision: differentiates interactions in set difference (and their count) from all other interactions (count 0)
# does not indicate if other interaction is in S intersection T or not in T


def setDifferenceCombinatorialCoverage(sourceCoverage, targetCoverage):
    """
    Computes the set difference of a target \ source data, given the combinatorial coverage
    computations of each.

    Parameters:
    - sourceCoverage: dict
        Combinatorial coverage results when computing on a source dataset.

    - targetCoverage: dict
        Combinatorial coverage results when computing on a target dataset.

    Returns:
    - setDifferenceStructure: dict
        Data structure containing set difference coverage info for each t, including the universe,
        appearing interactions in the set difference and their counts

    """

    interactionsInTarget = targetCoverage["countAppearingInteractions"]
    setDifferenceInteractions = 0  # those that appear in T but not S
    setDifferenceInteractionsCounts = []
    setDifferenceStructure = {
        "t": targetCoverage["t"],
        "k": targetCoverage["k"],
        "setDifferenceInteractionsCounts": setDifferenceInteractionsCounts,
    }
    for rank in range(0, len(targetCoverage["countsAllCombinations"])):
        LOGGER_COMBI.log(level=15, msg=targetCoverage["countsAllCombinations"][rank])

        counts = [
            0 for i in range(0, len(targetCoverage["countsAllCombinations"][rank]))
        ]
        for interaction in range(0, len(targetCoverage["countsAllCombinations"][rank])):
            if (
                targetCoverage["countsAllCombinations"][rank][interaction] > 0
                and sourceCoverage["countsAllCombinations"][rank][interaction] == 0
            ):
                counts[interaction] = 1
                setDifferenceInteractions += 1
        setDifferenceStructure["setDifferenceInteractionsCounts"].append(counts)
    setDifferenceStructure["interactionsInTarget"] = interactionsInTarget
    setDifferenceStructure["setDifferenceInteractions"] = setDifferenceInteractions

    LOGGER_COMBI.log(level=15, msg=setDifferenceStructure)

    return setDifferenceStructure


# computes the set difference of T \ S given the combinatorial coverage representation of both has already been computed
# ternary decision: if in set difference gives count, if in intersection count = 0, if not in T count = -1
def setDifferenceCombinatorialCoverageConstraints(sourceCoverage, targetCoverage):
    """
    Computes the set difference of a target \ source data, given the combinatorial coverage
    computations of each.

    Parameters:
    - sourceCoverage: dict
        Combinatorial coverage results when computing on a source dataset.

    - targetCoverage: dict
        Combinatorial coverage results when computing on a target dataset.

    Returns:
    - setDifferenceStructure: dict
        Data structure containing set difference coverage info for each t, including the universe,
        appearing interactions in the set difference and their counts

    """
    # NOTE: Difference between this and constraints?

    interactionsInTarget = targetCoverage["countAppearingInteractions"]
    setDifferenceInteractions = 0  # those that appear in T but not S
    setDifferenceInteractionsCounts = []
    setDifferenceStructure = {
        "t": targetCoverage["t"],
        "k": targetCoverage["k"],
        "setDifferenceInteractionsCounts": setDifferenceInteractionsCounts,
    }
    for rank in range(0, len(targetCoverage["countsAllCombinations"])):
        LOGGER_COMBI.log(level=15, msg=targetCoverage["countsAllCombinations"][rank])

        counts = [
            -1 for i in range(0, len(targetCoverage["countsAllCombinations"][rank]))
        ]
        for interaction in range(0, len(targetCoverage["countsAllCombinations"][rank])):
            if targetCoverage["countsAllCombinations"][rank][interaction] > 0:
                # in the target but not source
                if sourceCoverage["countsAllCombinations"][rank][interaction] == 0:
                    counts[interaction] = 1
                    setDifferenceInteractions += 1
                # in both source and target
                elif sourceCoverage["countsAllCombinations"][rank][interaction] > 0:
                    counts[interaction] = 0
            # EXPERIMENTAL 07-25-24
            elif targetCoverage["countsAllCombinations"][rank][interaction] == 0:
                if sourceCoverage["countsAllCombinations"][rank][interaction] == 0:
                    counts[interaction] = -1  # REVERTED FROM 2 05/16/25
                else:
                    assert counts[interaction] == -1
        setDifferenceStructure["setDifferenceInteractionsCounts"].append(counts)
    setDifferenceStructure["interactionsInTarget"] = interactionsInTarget
    setDifferenceStructure["setDifferenceInteractions"] = setDifferenceInteractions

    LOGGER_COMBI.log(
        level=15, msg="Set difference structure: {}".format(setDifferenceStructure)
    )

    return setDifferenceStructure


# computes the missing interactions
def computeMissingInteractions(representation, combinatorialCoverageStructure):
    """
    Computes and finds the interactions in the data theoretically possible that do not
    exist in the data.

    Parameters:
    - representation: Mapping
        Encoded representation of a dataset

    - combinatorialCoverageStructure: dict
        Combinatorial coverage results data structure to write CC metric to

    Returns:
    - missing: list
        A list of missing interactions
    """

    t = combinatorialCoverageStructure["t"]
    k = combinatorialCoverageStructure["k"]
    # either true k choose t or k' choose t' when label centric
    # previously computed correctly to store the combination counts so take list length
    kct = len(combinatorialCoverageStructure["countsAllCombinations"])
    tprime = t
    if label_centric:
        tprime -= 1  # require t-1 way interactions of all other columns

    LOGGER_COMBI.log(
        msg="t: {}, t': {}, k choose t: {}".format(t, tprime, kct), level=25
    )

    # stores missing combinations lists of tuples: one for each t-way interaction with (col,val) pair
    missing = []
    # create the empty list for counting, length kct for all ranks
    coveragePerRank = [0 for index in range(0, kct)]
    # go rank by rank through kct ranks
    for rank in range(0, kct):
        numinteractionsForRank = len(
            combinatorialCoverageStructure["countsAllCombinations"][rank]
        )
        countAppearingInteractionsInRank = 0
        # for each combination in a rank
        for index in range(0, numinteractionsForRank):
            # get the number of appearances for the combination at that index
            i = combinatorialCoverageStructure["countsAllCombinations"][rank][index]
            # if the number is at least 1, increment the appearing count
            if i > 0:
                countAppearingInteractionsInRank += 1
            # otherwise it's a missing combination
            else:
                # unranks and returns the set for the missing combination
                set = []
                tupleCols = abstraction.rank_to_tuple(set, rank, tprime)
                # the rank is for the t-1 cols not including the label col
                if label_centric:
                    set.append(k - 1)
                tupleValues = abstraction.convert_index_to_value_combo(
                    representation, index, set, t
                )
                interaction = []
                for element in range(0, t):
                    # pair is (col, val)
                    pair = (tupleCols[element], tupleValues[element])
                    # build the t-way interaction of pairs
                    interaction.append(pair)
                missing.append(interaction)
        coveragePerRank[rank] = (
            countAppearingInteractionsInRank / numinteractionsForRank
        )

    LOGGER_COMBI.log(msg="Missing interactions: {}".format(missing), level=25)

    return missing


# computes the interactions in the set difference (present in representation but not the other set)
def computeSetDifferenceInteractions(representation, setDifferenceCoverageStructure):
    """
    Computes the interactions in the set difference between a source and target dataset,
    interactions that are present in the represenation of the target and not the source.

    Parameters
    - representation: Mapping
        Encoded representation of a dataset

    - setDifferenceCoverageStructure: dict
        Computed set difference coverage structure ***

    Returns
    - setDifferenceInteractions: list
        A list of interactions appearing in the set difference
    """

    t = setDifferenceCoverageStructure["t"]
    k = setDifferenceCoverageStructure["k"]
    # find the set difference interactions
    # stores missing combinations lists of tuples: one for each t-way interaction with (col,val) pair
    setDifferenceInteractions = []
    # either true k choose t or k' choose t' when label centric
    kct = len(setDifferenceCoverageStructure["setDifferenceInteractionsCounts"])
    # previously computed correctly to store the combination counts so take list length
    tprime = t
    if label_centric:
        tprime -= 1  # require t-1 way interactions of all other columns
    for rank in range(0, kct):  # go rank by rank through kct ranks
        numinteractionsForRank = len(
            setDifferenceCoverageStructure["setDifferenceInteractionsCounts"][rank]
        )
        for index in range(0, numinteractionsForRank):  # for each combination in a rank
            # if the number is at least 1, it appears in the set difference
            if (
                setDifferenceCoverageStructure["setDifferenceInteractionsCounts"][rank][
                    index
                ]
                > 0
            ):
                # missing combination
                set = []
                # unranks and returns the set for the missing combination
                tupleCols = abstraction.rank_to_tuple(set, rank, tprime)
                # the rank is for the t-1 cols not including the label col
                if label_centric:
                    set.append(k - 1)
                tupleValues = abstraction.convert_index_to_value_combo(
                    representation, index, set, t
                )
                interaction = []
                for element in range(0, t):
                    # col, val pair
                    pair = (tupleCols[element], tupleValues[element])
                    # build the t-way interaction of pairs
                    interaction.append(pair)
                setDifferenceInteractions.append(interaction)
    return setDifferenceInteractions


# uses the representation structure to match on set difference interactions but then pulls the original row from the DF
# produces two files - one with the images that have any interaction in the set difference and the complement
# assumes the first column in the schema is an ID column and includes the substring 'ID'
def identifyImagesWithSetDifferenceInteractions(df, decodedSetDifference):
    # schema = df.schema.names
    if id not in df.columns:
        raise Exception("Identify Images requires dataframe to have Image ID")
    ids = []
    # decodedSetDifference is a list of lists; list of interactions
    # interactions are lists of column/value pairs (assignments)
    # for each interaction in set difference, filter the dataframe on the AND condition
    # there will be t of them so can't hardcode; try repeated filters
    # return the image IDs of records containing that interaction
    for interaction in decodedSetDifference:
        tempdf = df
        for assignment in interaction:
            # tempdf.filter(F.col(assignment[0]) == assignment[1])
            tempdf = tempdf.loc[tempdf[assignment[0]] == assignment[1]]
        # collect into one list
        smalldf = tempdf[id]  # tempdf.select(id)
        ids.extend(smalldf)  # ids.extend(smalldf.topd()[id].tolist())
    # keep only unique images
    setDiffImages = pd.DataFrame(ids, columns=[id]).drop_duplicates()
    return setDiffImages


def cc_dict(CC):
    jsondict = {}
    jsondict["count appearing interactions"] = CC["countAppearingInteractions"]
    jsondict["total possible interactions"] = CC["totalPossibleInteractions"]
    jsondict["CC"] = CC["countAppearingInteractions"] / CC["totalPossibleInteractions"]

    return jsondict


def sdcc_dict(SDCC):
    jsondict = {}
    jsondict["count interactions appearing in set"] = SDCC["interactionsInTarget"]
    jsondict["count interactions in set difference"] = SDCC["setDifferenceInteractions"]
    try:
        sdcc = SDCC["setDifferenceInteractions"] / SDCC["interactionsInTarget"]
    except:
        # NOTE: HOW TO HANDLE 0 sdcc[INTERACTIONS IN TARGET]
        sdcc = 0.0  # '0 Interactions in target'
    jsondict["SDCC"] = sdcc
    return jsondict


# -------------------- Functions Provide Control Flow --------------------#


def decodeCombinations(data, CC, t):
    # @leebri2n
    ranks = []
    k = CC["k"]
    kct = len(CC["countsAllCombinations"])
    tprime = t
    if label_centric:
        tprime -= 1  # require t-1 way interactions of all other columns
    # go rank by rank through kct ranks
    for rank in range(0, kct):
        numinteractionsForRank = len(CC["countsAllCombinations"][rank])
        countAppearingInteractionsInRank = 0
        for index in range(0, numinteractionsForRank):
            # unranks and returns the set
            set = []
            tupleCols = abstraction.rank_to_tuple(set, rank, tprime)
            # the rank is for the t-1 cols not including the label col
            if label_centric:
                set.append(k - 1)
            tupleValues = abstraction.convert_index_to_value_combo(data, index, set, t)
            interaction = []
            for element in range(0, t):
                # pair is (col, val)
                pair = (tupleCols[element], tupleValues[element])
                # build the t-way interaction of pair0s
                interaction.append(pair)
            decodedCombination = ""
            decodedInteraction = ""
            for pair in interaction:
                if decodedCombination == "":
                    decodedCombination += str(abstraction.decoding_combo(data, *(pair)))
                else:
                    decodedCombination += "*{}".format(
                        str(abstraction.decoding_combo(data, *(pair)))
                    )
                decodedInteraction += str(abstraction.decoding(data, *(pair)))

        if decodedCombination not in ranks:
            ranks.append(decodedCombination)

    return ranks


# -------------------- Experimental Code for Test Set Design --------------------#
def computeFrequencyInteractions(CC):
    appearancesList = []
    for rank in range(0, CC["kct"]):
        numInteractionsForRank = len(CC["countsAllCombinations"][rank])
        countAppearingInteractionsInRank = 0
        appearancesAllInteractions = sum(CC["countsAllCombinations"][rank])
        for index in range(0, numInteractionsForRank):
            appearancesThisInteraction = CC["countsAllCombinations"][rank][index]
            percentageOfAllInteractions = (
                float(appearancesThisInteraction) / appearancesAllInteractions
            )
            temp = [
                rank,
                index,
                appearancesThisInteraction,
                percentageOfAllInteractions,
            ]
            appearancesList.append(temp)
    return appearancesList


def goalSamples(CC, testSetSize):
    numSamples = []
    for rank in range(0, CC["kct"]):
        numSamples.append(testSetSize / len(CC["countsAllCombinations"][rank]))
    return numSamples


def frequencyInteractions(CC, goalSamples):
    data = CC["representation"]
    k = CC["k"]
    t = CC["t"]
    kprime = k
    tprime = t
    if label_centric:
        tprime -= 1  # require t-1 way interactions of all other columns
        kprime -= 1  # and only consider the first k-1 columns in the rank
    appearancesList = []

    print("Frequencies of interactions:")

    for rank in range(0, CC["kct"]):
        numInteractionsForRank = len(CC["countsAllCombinations"][rank])
        countAppearingInteractionsInRank = 0
        appearancesAllInteractions = sum(CC["countsAllCombinations"][rank])
        # for each combination in a rank
        for index in range(0, numInteractionsForRank):
            # get the number of appearances for the combination at that index
            appearancesThisInteraction = CC["countsAllCombinations"][rank][index]
            # if the number is at least 1, increment the appearing count
            if appearancesThisInteraction > 0:
                countAppearingInteractionsInRank += 1
            # unranks and returns the set for the interaction
            set = []
            tupleCols = abstraction.rank_to_tuple(set, rank, tprime)
            # the rank is for the t-1 cols not including the label col
            if label_centric:
                set.append(k - 1)
            tupleValues = abstraction.convert_index_to_value_combo(
                data, index, set, CC["t"]
            )
            interaction = []
            for element in range(0, CC["t"]):
                # pair is (col, val)
                pair = (tupleCols[element], tupleValues[element])
                # build the t-way interaction of pairs
                interaction.append(pair)
            percentageOfAllInteractions = (
                float(appearancesThisInteraction) / appearancesAllInteractions
            )
            temp = [
                rank,
                index,
                appearancesThisInteraction,
                percentageOfAllInteractions,
            ]
            appearancesList.append(temp)

            LOGGER_COMBI.log(
                msg="Rank {}, index {} appears {} times, frequency of {}.".format(
                    str(rank),
                    str(index),
                    appearancesThisInteraction,
                    round(percentageOfAllInteractions, 3),
                    abstraction.decode_interaction(data, interaction),
                ),
                level=15,
            )

            if appearancesThisInteraction < goalSamples[rank]:
                print(
                    abstraction.decode_interaction(data, interaction),
                    "INSUFFICIENT SAMPLES ",
                    appearancesThisInteraction,
                    " TO MEET TEST SET GOAL",
                    goalSamples[rank],
                )
            elif appearancesThisInteraction < (goalSamples[rank] * 2):
                print(
                    abstraction.decode_interaction(data, interaction),
                    "SAMPLE COUNT ",
                    appearancesThisInteraction,
                    " < 2x TEST SET GOAL",
                    goalSamples[rank],
                )

        LOGGER_COMBI.log(
            msg="Coverage for rank {} is {}".format(
                rank, countAppearingInteractionsInRank / numInteractionsForRank
            ),
            level=25,
        )
    return appearancesList


# takes in a row and whether the row is being added or removed and updates the coverage structure assuming the from the representation was added to the existing coverage


def updateCoverage(CC, row, add=True):
    countUpdate = 1 if add else -1
    countAppearingInteractions = CC["countAppearingInteractions"]
    countsAllCombinations = CC["countsAllCombinations"]
    k = CC["k"]
    t = CC["t"]
    representation = CC["representation"]
    kprime = k
    tprime = t
    if label_centric:  # if label_centric,
        tprime -= 1  # require t-1 way interactions of all other columns
        kprime -= 1  # and only consider the first k-1 columns in the rank
    kct = CC["kct"]
    for rank in range(0, kct):  # go rank by rank
        # enumerate all combinations
        set = []  # start with an empty set to pass through to the function
        # find the set of columns corresponding to this rank
        set = abstraction.rank_to_tuple(set, rank, tprime)
        interactionsForRank = 1  # compute v1*v2*...*vt so start with 1
        if label_centric:  # if every combination must include the label column
            set.append(kprime)  # add the last column (label) to the set
        # it's t not t' as every combination is of t columns regardless
        for column in range(0, t):
            interactionsForRank *= len(representation.values[set[column]])
        if interactionsForRank != len(CC["countsAllCombinations"][rank]):
            print("interactionsForRank error")
            exit(-1)
        index = abstraction.convert_value_combo_to_index(representation, set, row)
        # if verbose:
        #    symbols = abstraction.convert_index_to_value_combo(representation, index, set, t)
        #    print('row\t', row, '\t has symbols ', symbols, ' corresponding to index ', index)
        CC["countsAllCombinations"][rank][index] += countUpdate
        if (add and CC["countsAllCombinations"][rank][index] == 1) or (
            not add and CC["countsAllCombinations"][rank][index] == 0
        ):
            CC["countAppearingInteractions"] += countUpdate
        # if verbose:
        #    print('rank: ', rank, ', set:', set, ', interactions: ', interactionsForRank)


def modifyTest(ids, CC, row, add=True):
    updateCoverage(CC, row, add)
    if ids[row]["inTest"] == add:
        print(
            "error: request would not modify test set. row is already present or absent"
        )
        exit(-1)
    ids[row]["inTest"] = add
    return


def computeScoreIDs(IDs, CC, frequency):
    t = CC["t"]
    representation = CC["representation"]
    kprime = CC["k"]
    tprime = t
    if label_centric:  # if label_centric,
        tprime -= 1  # require t-1 way interactions of all other columns
        kprime -= 1  # and only consider the first k-1 columns in the rank
    kct = CC["kct"]

    for interaction in frequency:
        rank = interaction[0]
        index = interaction[1]
        f = interaction[3]
        set = []  # start with an empty set to pass through to the function
        # find the set of columns corresponding to this rank
        set = abstraction.rank_to_tuple(set, rank, tprime)
        symbols = abstraction.convert_index_to_value_combo(
            representation, index, set, t
        )
        # print("SYMBOLS\n\n\n", symbols, type(symbols))
        for row in range(0, len(representation.data)):
            match = True
            for i in range(0, t):
                col = set[i]
                if representation.data[row][col] != symbols[i]:
                    match = False
                    break
            if match:
                IDs[row]["score"] += f
                # print("rank: ", rank, "set: ", set, "symbols: ", symbols, "row: ", row, "score: ", IDs[row]['score'])
    return


def findSamples(IDs, test_cc, goal, sortedFrequency):
    t = test_cc["t"]
    representation = test_cc["representation"]
    kprime = test_cc["k"]
    tprime = t
    if label_centric:  # if label_centric,
        tprime -= 1  # require t-1 way interactions of all other columns
        kprime -= 1  # and only consider the first k-1 columns in the rank
    kct = test_cc["kct"]
    for interaction in sortedFrequency:
        rank = interaction[0]
        index = interaction[1]
        set = []  # start with an empty set to pass through to the function
        # find the set of columns corresponding to this rank
        set = abstraction.rank_to_tuple(set, rank, tprime)
        symbols = abstraction.convert_index_to_value_combo(
            representation, index, set, t
        )
        # find the rows that have the interaction we are interested in and sort them by their score
        matchingIDs = []
        for row in range(0, len(representation.data)):
            match = True
            for i in range(0, t):
                col = set[i]
                if representation.data[row][col] != symbols[i]:
                    match = False
                    break
            if match and not IDs[row]["inTest"]:
                # print(row)
                matchingIDs.append([row, IDs[row]["score"]])
        matchingIDs.sort(key=lambda x: x[1])
        # add a number of samples going from top of sorted list down until we get to the goal
        numSamplesToSelect = (
            int(math.ceil(goal[rank])) - test_cc["countsAllCombinations"][rank][index]
        )
        # print("numSamples: ", goal[rank], test_cc['countsAllCombinations'][rank][index], numSamplesToSelect)
        if numSamplesToSelect > len(matchingIDs):
            raise AssertionError("Too few samples to meet goal")
            exit(-1)
        for i in range(0, numSamplesToSelect):
            modifyTest(IDs, test_cc, matchingIDs[i][0], add=True)
            # print(matchingIDs[i][0], IDs[matchingIDs[i][0]])
    return


def testsetPostOptimization(IDs, test_cc, goal):
    t = test_cc["t"]
    representation = test_cc["representation"]
    kprime = test_cc["k"]
    tprime = t
    if label_centric:  # if label_centric,
        tprime -= 1  # require t-1 way interactions of all other columns
        kprime -= 1  # and only consider the first k-1 columns in the rank
    kct = test_cc["kct"]
    redundancy = []
    for rank in range(0, kct):  # For each rank
        r = [rank, 0]  # rank: int, r: list, i: list
        i = []
        # For each interaction
        for interaction in range(0, len(test_cc["countsAllCombinations"][rank])):
            # If number of interactions/goal for this combo > 0?
            if test_cc["countsAllCombinations"][rank][interaction] / goal[rank] > r[1]:
                # Update [1] to be that proportion. Stores where there might be imbalance?
                r[1] = test_cc["countsAllCombinations"][rank][interaction] / goal[rank]
            # print(goal[rank], test_cc['countsAllCombinations'][rank][interaction], r)
            # In any case,  add interaction, number of appearences
            i.append([interaction, test_cc["countsAllCombinations"][rank][interaction]])
        r.append(i)
        redundancy.append(r)

    LOGGER_COMBI.log(
        msg="Interaction redundancy list:\n{}{}".format(redundancy, type(redundancy)),
        level=15,
    )
    redundancy.sort(key=lambda x: x[1], reverse=True)
    LOGGER_COMBI.log(
        msg="Interaction redundancy list, sorted:\n{}{}".format(
            redundancy, type(redundancy)
        ),
        level=15,
    )

    for r in redundancy:
        rank = r[0]
        interactions = r[2]
        interactions.sort(key=lambda x: x[1], reverse=True)

        LOGGER_COMBI.log(
            msg="Sorted list for finding redundant coverage:\nRank: {}, Interactions: {}".format(
                rank, interactions
            ),
            level=15,
        )

        for i in interactions:
            index = i[0]
            set = []  # start with an empty set to pass through to the function
            # find the set of columns corresponding to this rank
            set = abstraction.rank_to_tuple(set, rank, tprime)
            symbols = abstraction.convert_index_to_value_combo(
                representation, index, set, t
            )
            # find the rows that have the interaction we are interested in
            for row in range(0, len(representation.data)):
                if IDs[row]["inTest"]:
                    match = True
                    for i in range(0, t):
                        col = set[i]
                        if representation.data[row][col] != symbols[i]:
                            match = False
                            break
                    if match:
                        LOGGER_COMBI.log(
                            level=15,
                            msg="Rank {}-index {} match on row {}".format(
                                rank, index, row
                            ),
                        )

                        # determine if removing this row would drop any interaction below the goal for interaction coverage in test
                        remove = True
                        for rank2 in range(0, kct):  # go rank by rank
                            set2 = []  # start with an empty set to pass through to the function
                            # find the set of columns corresponding to this rank
                            set2 = abstraction.rank_to_tuple(set2, rank2, tprime)
                            index2 = abstraction.convert_value_combo_to_index(
                                representation, set2, row
                            )

                            #    symbols = abstraction.convert_index_to_value_combo(representation, index2, set2, t)
                            LOGGER_COMBI.log(
                                level=15,
                                msg="Row \t{}\t has symbols: {} corresponding to index {}".format(
                                    row, symbols, index2
                                ),
                            )
                            LOGGER_COMBI.log(
                                level=15,
                                msg="Row \t{}\t has symbols: {} corresponding to index {}".format(
                                    row, symbols, index2
                                ),
                            )

                            if (
                                test_cc["countsAllCombinations"][rank2][index2]
                                <= goal[rank2]
                            ):
                                remove = False
                                break
                        if remove:
                            modifyTest(IDs, test_cc, row, add=False)

                            LOGGER_COMBI.log(
                                level=15,
                                msg="Removing row {}, {}".format(
                                    row, representation.data[row]
                                ),
                            )
                            LOGGER_COMBI.log(
                                level=15, msg="Resultant test CC: {}".format(test_cc)
                            )
    return


# TODO: rename to SIE and split out functionality that is just balanced test set construction vs. entire SIE (train/val)
def balanced_test_set(
    dataset_df: pd.DataFrame,
    dataset_name: str,
    sample_id_col,
    universe,
    strengths,
    test_set_size_goal,
    output_dir,
    include_baseline=True,
    construct_splits=False,
):
    split_dir_json = output.create_output_dir(os.path.join(output_dir, "splits_json"))
    split_dir_csv = output.create_output_dir(os.path.join(output_dir, "splits_csv"))

    # produce a random ordering of the data prior to representation creation so multiple runs produce different candidate results
    mMap = Mapping(universe["features"], universe["levels"], None)
    dataset_df = dataset_df.sample(len(dataset_df)).reset_index(drop=True)
    data_representation = abstraction.encoding(dataset_df, mMap, True)

    LOGGER_COMBI.log(msg="Metadata level map:\n{}".format(mMap), level=15)
    LOGGER_COMBI.log(
        msg="Data representation:\n{}".format(data_representation), level=15
    )
    LOGGER_COMBI.log(msg="DataFrame:\n{}".format(dataset_df), level=15)

    k = len(data_representation.features)
    t = max(strengths)  # maximum t
    if t > k:
        print("t =", t, " cannot be greater than number of features k =", k)
        return

    # computes CC for one dataset
    CC = combinatorialCoverage(data_representation, t)
    LOGGER_COMBI.log(level=25, msg="CC: {}".format(CC))

    decodedMissing = abstraction.decode_missing_interactions(
        data_representation, computeMissingInteractions(data_representation, CC)
    )
    output.writeCCtToFile(output_dir, dataset_name, t, CC)
    output.writeMissingtoFile(output_dir, dataset_name, t, decodedMissing)

    # build test set in 2 phases
    # phase 1: add samples to test set with first priority to achieve coverage
    # and second priority to have balance
    # sort the interactions by rareness and go through them one at a time
    # for each interaction, select samples to cover the interaction at the
    # desired target and update coverage structure
    # to prioritize balance, score each sample by rareness of interactions contained
    # phase 2: post-optimization
    # determine if a sample can be removed due to redundancy (over coverage)

    test_cc = copy.deepcopy(CC)
    for combinations in test_cc["countsAllCombinations"]:
        for interaction in range(0, len(combinations)):
            combinations[interaction] = 0
    test_cc["countAppearingInteractions"] = 0

    goal = goalSamples(CC, test_set_size_goal)
    print("\nGoal # samples per rank in test set:\n", goal, "\n")
    frequency = frequencyInteractions(CC, goal)
    # print("\nFrequency of interactions: ", frequency)
    sortedFrequency = copy.deepcopy(frequency)
    sortedFrequency.sort(key=lambda x: x[3])

    LOGGER_COMBI.log(
        level=25, msg="\nGoal # samples per rank in test set: {}\n".format(goal)
    )
    LOGGER_COMBI.log(
        level=25,
        msg="Sorted number and frequency of interactions: {}".format(sortedFrequency),
    )

    # create a data structure that maps sample IDs from data to the index in the data representation of encoded
    # features of the sample as well as whether the sample has been added to the Test set
    # and give the sample a score that is the sum of interaction frequencies contained in the sample
    idlist = list(dataset_df[sample_id_col])
    IDs = dict()  # Empty dict
    for index in range(0, len(data_representation.data)):
        IDs[index] = {"id": idlist[index], "inTest": False, "score": 0}

    # Pass 1 build the test set
    computeScoreIDs(IDs, CC, frequency)
    findSamples(IDs, test_cc, goal, sortedFrequency)
    print("\nINITIAL TESTCC\n", test_cc)
    # print("TEST", json.dumps(CC, indent=2))
    # Pass 2 reduce the test set
    testsetPostOptimization(IDs, test_cc, goal)
    # print("TEST", json.dumps(IDs, indent=2))

    # determine the max training set size m as the minimum over all withheld interactions
    allsamplesize = sum(CC["countsAllCombinations"][0])
    m = allsamplesize
    testsamplesize = sum(test_cc["countsAllCombinations"][0])
    kct = len(CC["countsAllCombinations"])
    print("\nRank, index, potential training sizes:")
    for rank in range(0, kct):  # go rank by rank
        for i in range(0, len(test_cc["countsAllCombinations"][rank])):
            samplesnotintest = (
                CC["countsAllCombinations"][rank][i]
                - test_cc["countsAllCombinations"][rank][i]
            )
            maxPoolForInteraction = allsamplesize - testsamplesize - samplesnotintest

            LOGGER_COMBI.log(
                level=25,
                msg="Rank {}, i {}, maximum pool size for interaction: {}".format(
                    rank, i, maxPoolForInteraction
                ),
            )

            m = min(m, maxPoolForInteraction)
            print("!!!", m)

    LOGGER_COMBI.log(level=15, msg="\nAll samples: {}".format(allsamplesize))
    LOGGER_COMBI.log(level=15, msg="Number of test samples: {}".format(testsamplesize))
    LOGGER_COMBI.log(level=15, msg="\nTraining set size: {}".format(m))

    # construct the sets
    test = []
    trainpool = []
    for index in IDs.keys():
        test.append(IDs[index]["id"]) if IDs[index]["inTest"] else trainpool.append(
            IDs[index]["id"]
        )
    # get all of the samples not in the test set for selection for the train sets
    querystring = sample_id_col + " in @trainpool"
    trainpool_df = dataset_df.query(querystring).reset_index(drop=True)

    """trainpoolDFrepresentation = abstraction.encoding(trainpool_df, mMap, True)
    trainpoolCC = combinatorialCoverage(trainpoolDFrepresentation, t)
    if trainpoolCC['totalPossibleInteractions'] !=trainpoolCC['countAppearingInteractions']:
        print('training pool does not contain some interaction. Exiting.')
        print(trainpoolCC)
        exit()"""

    # get all of the samples in the test set for later splitting into included/excluded subsets
    querystring = sample_id_col + " in @test"
    unitest_df = dataset_df.query(querystring)
    unitest_df.to_csv(os.path.join(output_dir, "test.csv"))
    trainpool_df.to_csv(os.path.join(output_dir, "trainpool.csv"))

    LOGGER_COMBI.log(level=15, msg="TRAINING POOL DF:\n{}".format(trainpool_df))
    LOGGER_COMBI.log(level=15, msg="UNIVERSAL TEST SET DF:\n{}".format(unitest_df))

    jsondict = cc_dict(CC)
    jsondict["max_t"] = t
    jsondict["max_training_pool_size"] = m
    jsondict["universal_test_set_size"] = testsamplesize

    # Moved 12.10.24
    jsondict["combination counts, all"] = CC["countsAllCombinations"]
    jsondict["combination counts, test"] = test_cc["countsAllCombinations"]

    if not construct_splits:
        return jsondict

    # TODO: up until now is test creation, from here is remainder

    jsondict["ids"] = {}
    jsondict["ids"]["test"] = test
    jsondict["ids"]["train_pool"] = trainpool

    # TODO: update from feature to interaction, iterate through ranks
    # for each interaction in each combination make the list of train values
    # these are samples that are not in the test set and do not have the withheld interaction
    # of the remaining options, randomly select m of them
    # in the future, select m that balance the rest of the interactions as best as possible
    for f in range(len(data_representation.features)):
        feature = data_representation.features[f]
        for v in range(len(data_representation.values[f])):
            value = str(data_representation.values[f][v])
            model_num = f"f{f}_i{v}_"

            print(data_representation.features[f], "excluding ", value)

            # future algorithms should build intelligently instead of relying on random chance for coverage
            # but for now, if it's possible, resample until we have coverage (could be a very long time)
            # impossibility check - if the training pool does not cover all interactions aside from the withheld one
            # then it doesn't matter how many times we resample, the sampled training set will not cover all
            coveredpossible = True
            trainpool_df_model = trainpool_df.loc[trainpool_df[feature] == value]
            assert type(trainpool_df_model) is pd.DataFrame
            trainpool_df_model = trainpool_df_model.reset_index(drop=True)

            print("Trainpool dim", trainpool_df_model.shape)
            trainpool_df_model_representation = abstraction.encoding(
                trainpool_df_model, mMap, True
            )
            trainpool_cc_model = combinatorialCoverage(
                trainpool_df_model_representation, t
            )

            LOGGER_COMBI.log(
                level=15,
                msg="Training pool model counts: {}".format(
                    trainpool_cc_model["countsAllCombinations"]
                ),
            )

            # TODO: update from feature to interaction
            for rank in range(0, kct):  # go rank by rank
                for i in range(0, len(test_cc["countsAllCombinations"][rank])):
                    if (
                        rank != f
                        and i != v
                        and trainpool_cc_model["countsAllCombinations"][rank][i] == 0
                    ):
                        coveredimpossible = (
                            True  # found a 0 that isn't the withheld interaction
                        )
                        m = len(trainpool_df_model)  # EXP ADDED 12.03.2024
                        print(
                            "Warning: Model's Training Pool does not cover all other interactions, so constructed train can't either."
                        )

            while True:  # execute at least once no matter what
                # sample from whole train pool
                train_df = trainpool_df_model.sample(m).reset_index(drop=True)

                LOGGER_COMBI.log(
                    level=15,
                    msg="Number of samples containing {} in training set: {}".format(
                        value, len(train_df[train_df[feature] == value])
                    ),
                )
                LOGGER_COMBI.log(
                    level=25, msg="Training set:\n{}".format(train_df.to_string())
                )

                # check that constructed training set covers everything EXCEPT the withheld interaction
                train_df_representation = abstraction.encoding(train_df, mMap, True)
                train_cc = combinatorialCoverage(train_df_representation, t)

                LOGGER_COMBI.log(
                    level=25,
                    msg="Train CC counts {}".format(train_cc["countsAllCombinations"]),
                )

                # TODO: update from feature to interaction
                covered = True  # REFERNCING AFTER ASSIGNMENT BC UNBOUND LOCAL ERROR THIS MIGHT CAUSE ERRORS
                for rank in range(0, kct):  # go rank by rank
                    for i in range(0, len(test_cc["countsAllCombinations"][rank])):
                        if (
                            rank != f
                            and i != v
                            and train_cc["countsAllCombinations"][rank][i] == 0
                        ):
                            covered = False
                            print(
                                "Constructed training set does not cover all other interactions. Will sample again."
                            )
                if covered or coveredimpossible:
                    break

            LOGGER_COMBI.log(
                level=15,
                msg="\n Model {} excludes interaction {}, {}".format(
                    model_num, feature, value
                ),
            )

            cut = math.ceil(m * 0.8)
            test_df_exclude = unitest_df.loc[unitest_df[feature] == value]
            test_df_include = unitest_df.loc[unitest_df[feature] != value]

            train_ids = list(train_df[sample_id_col])
            include_test_ids = list(test_df_include[sample_id_col])
            exclude_test_ids = list(test_df_exclude[sample_id_col])
            jsondict["ids"][f"model_{model_num}"] = {
                "test_included": include_test_ids,
                "test_excluded": exclude_test_ids,
                "train": train_ids[:cut],
                "validation": train_ids[cut:],
            }

            train_df.to_csv(os.path.join(split_dir_csv, f"train_{model_num}-t{t}.csv"))
            exclude_test_ids.to_csv(
                os.path.join(split_dir_csv, f"test_{model_num}-excl_trn-t{t}.csv")
            )
            include_test_ids.to_csv(
                os.path.join(split_dir_csv, f"test_{model_num}-incl_trn-t{t}.csv")
            )

    if include_baseline:
        cut = math.ceil(m * 0.8)
        train_df = trainpool_df.sample(m)  # , random_state=baseline_seed)

        train_ids = list(train_df[sample_id_col])
        include_test_ids = list(unitest_df[sample_id_col])

        jsondict["ids"]["model_x_"] = {
            "test_included": include_test_ids,
            "test_excluded": [],
            "train": train_ids[:cut],
            "validation": train_ids[cut:],
        }

    # Per split JSON
    for key in jsondict["ids"].keys():
        assert "model_" in key
        output.output_json_readable(
            jsondict["ids"][key],
            write_json=True,
            file_path=os.path.join(split_dir_json, f"{key}.json"),
        )

    return jsondict

# Requires a representation to be completed and a performance dictionary where
# name of key is performance metric and value is a list where each index is the performance for a sample
# it must have the same index as the representation so we can go row by row
# Time increase to compute coverage and performance by interaction simultaneously is minimal, just an extra index and increment per interaction
# Doing these sequentially requires 2 passes through the representation, so N * number of interactions
# Storage may be a factor
# Future versions could decouple these activities in an alternative function


def computePerformanceByInteraction(
    representation: Mapping, t: int, performance_df: pd.DataFrame, test=True
):
    totalPossibleInteractions = 0
    countAppearingInteractions = 0
    countsAllCombinations = []
    # print(type(representation.data), representation.data.shape)
    k = len(representation.features)  # number of columns in data
    coverageDataStructure = {
        "t": t,
        "k": k,
        "representation": representation,
        "countsAllCombinations": countsAllCombinations,
    }
    kprime = k
    tprime = t
    if label_centric:  # if label_centric,
        tprime -= 1  # require t-1 way interactions of all other columns
        kprime -= 1  # and only consider the first k-1 columns in the rank
    # k choose t - number of  combinations of t columns
    kct = comb(kprime, tprime, exact=True)
    metrics = performance_df.columns

    LOGGER_COMBI.log(level=15, msg="PERFORMANCE DF:\n{}".format(performance_df))
    performanceDataStructure = {
        metric: [0] * kct for metric in metrics
    }  # dict.fromkeys(metrics)
    # for metric in metrics:
    #    performanceDataStructure[metric] =

    LOGGER_COMBI.log(
        level=25,
        msg="k: {}, k': {}, t: {}, t': {}, k choose t: {}".format(
            k, kprime, t, tprime, kct
        ),
    )
    coverageDataStructure = {
        "subset": None,
        "t": t,
        "k": k,
        "kct": kct,
        "representation": representation,
        "countsAllCombinations": countsAllCombinations,
    }
    coverageDataStructure["subset"] = "test" if test else "train"

    for rank in range(0, kct):  # go rank by rank
        # enumerate all combinations
        set = []  # start with an empty set to pass through to the function
        # find the set of columns corresponding to this rank
        set = abstraction.rank_to_tuple(set, rank, tprime)
        interactionsForRank = 1  # compute v1*v2*...*vt so start with 1
        if label_centric:  # if every combination must include the label column
            set.append(kprime)  # add the last column (label) to the set
        # it's t not t' as every combination is of t columns regardless
        for column in range(0, t):
            interactionsForRank *= len(representation.values[set[column]])
        totalPossibleInteractions += interactionsForRank  # update the total
        # countsEachCombination COULD be boolean BUT we want to get all counts for later diversity metrics
        countsEachCombination = [
            0 for index in range(0, interactionsForRank)
        ]  # create list of 0s for counting
        for metric in metrics:
            performanceDataStructure[metric][rank] = [
                0 for index in range(0, interactionsForRank)
            ]  # create list of 0s for aggregating performance
        # count the combinations that appear in this rank by checking all of the rows in the columns indicated by set[]
        for row in range(0, len(representation.data)):
            index = abstraction.convert_value_combo_to_index(representation, set, row)
            # if verbose:
            #    print("rank ", rank, " set ", set, " row ", row, " index ", index)
            # here we update the performance for the interaction by multiplying by the current count to get current performance
            # adding performance of the row then dividing by the new count
            currentCount = countsEachCombination[index]
            for metric in metrics:
                rowPerformance = performance_df[metric].iloc[row]
                if not math.isnan(rowPerformance):  # PERFORMANCE HERE
                    currentPerformance = performanceDataStructure[metric][rank][index]
                    performanceDataStructure[metric][rank][index] = (
                        (currentPerformance * currentCount) + rowPerformance
                    ) / (currentCount + 1)
            # last update the count in the coverage structure
            countsEachCombination[index] += 1
        countsAllCombinations.append(countsEachCombination)
        # update the count -- since might be more than 0, count if a cell is nonzero rather than summing the counts
        for index in range(0, interactionsForRank):
            if countsEachCombination[index] > 0:
                countAppearingInteractions += 1
            if countsEachCombination[index] == 0:
                for metric in metrics:
                    performanceDataStructure[metric][rank][index] = None
    coverageDataStructure["totalPossibleInteractions"] = totalPossibleInteractions
    coverageDataStructure["countAppearingInteractions"] = countAppearingInteractions

    LOGGER_COMBI.log(
        level=25, msg="PERFORMANCE Data Structure:\n{}".format(performance_df)
    )

    return coverageDataStructure, performanceDataStructure
