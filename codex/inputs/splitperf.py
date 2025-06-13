import json
import os


def extract_split_simple(codex_input: dict) -> tuple[dict | list[dict], str]:
    codex_dir = codex_input["codex_dir"]

    split_dir = codex_input.get("split_dir", "")
    split_filename = codex_input.get("split_filename")

    split_dir = os.path.realpath(os.path.join(codex_dir, split_dir))

    if split_filename is None:
        split = None
        split_id = None
    else:
        split, split_id = __extract_list_of_dicts(split_dir, split_filename, "split")

    return split, split_id


def extract_perf_simple(codex_input: dict) -> tuple[dict | list[dict], str, list]:
    codex_dir = codex_input["codex_dir"]

    perf_dir = codex_input.get("performance_dir", "")
    perf_filename = codex_input.get("performance_filename")

    perf_dir = os.path.realpath(os.path.join(codex_dir, perf_dir))

    if perf_filename is None:
        perf = None
        split_id_perf = None
        metrics = None
    else:
        perf, split_id_perf = __extract_list_of_dicts(
            perf_dir, perf_filename, "performance"
        )

        metrics = codex_input.get("metrics", [])

    return perf, split_id_perf, metrics


def __extract_list_of_dicts(dir: str, filename: str | list, mode: str):
    info_dicts = None
    info_ids = None
    if type(filename) is str:
        with open(os.path.join(dir, filename)) as f:
            info_dicts = json.load(f)
            info_ids = __extract_split_id_type("split", info_dicts, filename)
    elif type(filename) is list:
        info_dicts = []
        info_ids = []

        for filename_i in filename:
            with open(os.path.join(dir, filename_i)) as f:
                info_dict_i = json.load(f)
                info_dicts.append(info_dict_i)
                info_ids.append(
                    __extract_split_id_type("performance", info_dict_i, filename_i)
                )
    else:
        raise TypeError()

    return info_dicts


def __extract_split_id_type(mode: str, container: dict, filename: str):
    if mode == "split":
        split_id = container.get("split_id", filename)

    elif mode == "performance":
        split_id = container.get("split_id", "No split ID provided.")

    return split_id


def extract_sp(codex_input, split_filename=None, performance_filename=None):
    """
    Extracts one or more split and performance files from input specifications.

    Returns:
    split: dict
        Read from a JSON file whose keys are the split partition and values are lists of
        the sample ID's as presented in the dataset. Or, if given as a list of split files,
        nested under the split file name it came from.

    performance: dict
        Read from a JSON file containing a model's performance on a test set. Or, if given as a
        list of split files, is nested under the split file name it came from.

    metric: str
        Chosen metric to evaluate performance for the experiment.
    """
    codex_dir = codex_input["codex_dir"]
    split_folder = codex_input["split_dir"]
    performance_folder = codex_input["performance_dir"]
    metric = codex_input["metric"]

    # Initial pass
    if split_filename is None and performance_filename is None:
        split_filename = codex_input["split_filename"]
        performance_filename = codex_input["performance_filename"]

    # Might be list if multiple, str if single
    if type(split_filename) is list:
        if performance_filename is None:
            performance_filename = [None] * len(split_filename)

        if type(performance_filename) is not list:
            raise ValueError("No corresponding performance files to given split files.")
        else:
            assert len(split_filename) == len(performance_filename)

        num_splits = len(split_filename)

        split = {filename: None for filename in split_filename}
        performance = {filename: None for filename in split_filename}

        for i in range(num_splits):
            split[split_filename[i]], performance[split_filename[i]], metric = (
                extract_sp(codex_input, split_filename[i], performance_filename[i])
            )

    elif type(split_filename) is str:
        # Add/return split
        # os.path.join(codex_dir, split_folder, split_filename)) as s:
        with open(
            os.path.abspath(os.path.join(codex_dir, split_folder, split_filename))
        ) as s:
            split = json.load(s)
            split["split_id"] = split_filename

        if performance_filename is None:
            performance = None
        else:
            with open(
                os.path.abspath(
                    os.path.join(codex_dir, performance_folder, performance_filename)
                )
            ) as p:
                performance = json.load(p)
                performance["split_id"] = split_filename

    elif split_filename is None and performance_filename is None:
        split = None
        performance = None
        metric = None

    else:
        raise ValueError("Unknown object for split file.")

    return split, performance, metric
