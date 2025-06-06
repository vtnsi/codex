import utils.config as config

MIN_REQ = [
    "mode",
    "codex_dir",
    "config_id",
    "output_dir",
    "dataset_dir",
    "dataset_filename",
    "features",
    "binning_dir",
    "binning_filename",
    "sample_id_column",
    "t",
]

OPT_REQ = {
    "universe_save_dir": None,
    "universe_filename": None,
    "use_augmented_universe": False,
    "counting_mode": "label_exclusive",
    "timestamp": False,
}


def input_checks(codex_input: dict, check_split=False, check_perf=False):
    __checks_generic(codex_input=codex_input)

    if check_split:
        __checks_split(codex_input)
    if check_perf:
        __checks_perf(codex_input)

    return True


def __checks_generic(codex_input: dict):
    for field in MIN_REQ:
        try:
            assert field in codex_input.keys()
        except:
            raise KeyError(
                f"Minimum requirement <{field}> not included in config file."
            )


def __checks_split(codex_input):
    split_fields = [
        "split_dir",
        "split_filename",
    ]

    for field in split_fields:
        try:
            assert field in codex_input.keys()
        except:
            raise KeyError(
                f"{codex_input['mode']} requires split info. <{field}> not included in config file."
            )


def __checks_perf(codex_input):
    perf_fields = ["performance_dir", "performance_filename", "metric"]

    for field in perf_fields:
        try:
            assert field in codex_input.keys()
        except:
            raise KeyError(
                f"{codex_input['mode']} requires performance metric info. <{field}> not included in config file."
            )

    return


def def_default_values(codex_input):
    for field in OPT_REQ:
        if field not in codex_input:
            codex_input[field] = OPT_REQ[field]

    return codex_input


def dse_prereq_check(codex_input):
    split, performance, metric = config.extract_sp(codex_input)

    if split is None:
        raise ValueError("No split file given.")

    # Given in split file
    try:
        source_name = codex_input["source_name"]
        if source_name is None:
            raise NameError
    except:
        source_name = "train"
    try:
        target_name = codex_input["target_name"]
        if target_name is None:
            raise NameError
    except:
        target_name = "test"

    # Exists check
    try:
        assert source_name in split
    except:
        raise KeyError(
            "Source split name {} does not exist in the split file. Specify correctly in the input file or correct the naming in the split file.".format(
                source_name
            )
        )
    try:
        assert target_name in split
    except:
        raise KeyError(
            "Target split name {} does not exist in the split file. Specify correctly in the input file or correct the naming in the split file.".format(
                target_name
            )
        )

    # No overlap check
    try:
        source_ids = split[source_name]
        target_ids = split[target_name]
        intersect = list(set(source_ids) & set(target_ids))
        assert len(intersect) == 0
    except:
        raise ValueError(
            "Split components {} and {} share samples.".format(source_name, target_name)
        )

    return source_name, target_name


def dsc_prereq_check(codex_input):
    split, performance, metric = config.extract_sp(codex_input)
    try:
        assert len(split) == len(performance)
    except:
        raise IndexError(
            "Ensure that each split file has their own corresponding performance file."
        )
