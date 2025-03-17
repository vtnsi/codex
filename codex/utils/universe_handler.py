import os
import logging
import copy
import pandas as pd

from ..modules import binning

def extract_metadataset(codex_input, dataset_path):
    """
    Given a decided dataset path, returns the dataset as a DataFrame.
    """
    features = codex_input["features"]

    dataset_df = pd.read_csv(dataset_path)
    dataset_df = dataset_df.map(str)

    keep = codex_input["features"].copy()
    keep.append(codex_input["sample_id_column"])
    drop = [col for col in list(dataset_df.columns) if col not in keep]
    dataset_df = dataset_df.drop(columns=drop)

    return dataset_df, features


def __learn_universe__(provided_universe, dataset_df, features):
    # combinatorial expects a dict with a list of features and a list of lists of feature levels
    # using the lists allows for positional indexing for rank/unrank conversion to mixed radix number
    # s.t. each interaction is represented as a mixed radix number for easy indexing

    # no universe provided so grab the features and learn the levels from the data

    if provided_universe is None:
        universe = {
            "features": features,
            "levels": [list(dataset_df[feature].unique()) for feature in features],
        }
    else:
        # a possibly partial universe provided. Add on any features we don't have yet and append new levels
        universe = copy.deepcopy(provided_universe)
        for i in range(0, len(features)):
            if features[i] not in universe["features"]:
                universe["features"].insert(i, features[i])
                universe["levels"].insert(i, [])
        for i in range(0, len(universe["features"])):
            feature = universe["features"][i]
            levels = list(dataset_df[feature].unique())
            for l in levels:
                if l not in universe["levels"][i]:
                    universe["levels"][i].append(l)

        for i, feature_levels in enumerate(universe["levels"]):
            if len(feature_levels) >= 25:
                logging.getLogger(__name__).warning(
                    "Over-abundance of levels in the dataset for {}. Consider if this feature requires a binning file.".format(
                        universe["features"][i]
                    )
                )

    assert universe is not None
    return universe


def initialize_universe(codex_input):
    """
    Handles the provided universe to learn or augment from. If neither bin or pre-existing
    universe exists, return NoneType. If binning file is used, used binned dataset moving
    forward and form universe from the specified bins. If preexisting universe is given,
    automatically use that.
    """
    data_dir = codex_input['data_directory']
    dataset_filename = codex_input['dataset_file']
    codex_dir = codex_input['codex_directory']

    try:
        dataset_path = os.path.join(data_dir, dataset_filename)
        print("DATASET PATH:", dataset_path, os.path.exists(dataset_path))

    except RuntimeError:
        print("Dataset directory not found... Executing with realpath.")
        dataset_path = os.path.realpath(os.path.join(codex_dir, data_dir, dataset_filename))
        print("DATASET PATH:", dataset_path, os.path.exists(dataset_path))
    provided_universe = None

    if codex_input["bin_file"] is not None:
        try:
            bin_directory = codex_input["bin_directory"]
            assert bin_directory is not None
        except:
            print("No bin_directory found.")
            bin_directory = ""
        bin_path = os.path.abspath(os.path.join(
            codex_input["codex_directory"], bin_directory, codex_input["bin_file"]
        ))
        provided_universe, dataset_path = binning.binfile(dataset_path, bin_path, codex_input['features'])

    if codex_input["universe"] is not None and type(codex_input["universe"]) is dict:
        print("Universe provided")
        logging.getLogger(__name__).info(
            "Universe provided via input file: {}".format("codex_input['universe']")
        )
        """with open(codex_input['universe']) as f:
            provided_universe = json.load(f)   """
        provided_universe = codex_input["universe"]

    return provided_universe, dataset_path


def get_universe_difference(dict_reference, dict_comparator):  # in, data
    """
    Gets the universe difference between a reference universe and one to
    be compared to.

    Usage: dict_reference typically the one expected to be larger.
    """
    uni_ref = {
        feature: dict_reference["levels"][i]
        for i, feature in enumerate(dict_reference["features"])
    }
    uni_comp = {
        feature: dict_comparator["levels"][i]
        for i, feature in enumerate(dict_comparator["features"])
    }

    uni_difference = {}
    uni_combination = copy.deepcopy(uni_comp)
    for i, feature in enumerate(uni_ref):
        if feature not in uni_comp:
            uni_combination[feature] = uni_ref[feature]
            uni_difference[feature] = uni_ref[feature]
        else:
            for j, level in enumerate(uni_ref[feature]):
                if level not in uni_comp[feature]:
                    uni_combination[feature].append(level)
                    if feature not in uni_difference:
                        uni_difference[feature] = []
                    uni_difference[feature].append(level)

    uni_result_difference = {"features": None, "levels": None}
    uni_result_augmented = {"features": None, "levels": None}

    uni_result_difference["features"] = list(uni_difference.keys())
    uni_result_difference["levels"] = [
        uni_difference[feature] for feature in uni_difference
    ]
    uni_result_augmented["features"] = list(uni_combination.keys())
    uni_result_augmented["levels"] = [
        uni_combination[feature] for feature in uni_combination
    ]

    return uni_result_difference, uni_result_augmented


def check_provided_subset(provided_universe, learned_universe):
    """
    Confirms if a universe is a subset of another.
    """
    difference, augmented = get_universe_difference(provided_universe, learned_universe)
    assert augmented == learned_universe

    if len(difference["features"]) == 0 and len(difference["levels"]) == 0:
        return True
    else:
        return False


def compare_universe_candidates(
    provided_universe, learned_universe, forceuse_augmented=False
):
    """
    Compares two universes.
    """
    if provided_universe is None or forceuse_augmented:
        return learned_universe

    assert check_provided_subset(provided_universe, learned_universe)

    return provided_universe


def define_input_space(codex_input, forceuse_augmented=False):
    """
    Handles the universe decision flow.

    Scenarios:
    - Provided binning file, no provided universe: Use binned universe
    - Provided binning file, provided universe: Use provided universe
    - No provided binning file, provided universe: Use provided universe
    - No provided binning file, no provided universe: Learn from data
    - Force augmented universe: Use provided universe as well as additional
    elements learned from data not present in provided universe.
    """

    provided_universe, dataset_path = initialize_universe(codex_input)
    if "Binned" in dataset_path:
        logging.getLogger(__name__ + ".binning").info(
            "Binning file provided. Opting to using binned dataset. Universe provided via bin file: {}.".format(
                codex_input["bin_file"]
            )
        )
    else:
        logging.getLogger(__name__ + ".binning").info(
            "No binning file found; opting for unbinned dataset"
        )

    dataset_df, features = extract_metadataset(codex_input, dataset_path)
    learned_universe = __learn_universe__(provided_universe, dataset_df, features)
    universe = compare_universe_candidates(
        provided_universe, learned_universe, forceuse_augmented=forceuse_augmented
    )
    logging.getLogger("{}_universe".format(__name__)).info(universe)

    return universe, dataset_df
