import os

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

    initialize_default_optional(codex_input)

    return codex_input


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


def initialize_default_optional(codex_input):
    for field in OPT_REQ:
        if field not in codex_input:
            codex_input[field] = OPT_REQ[field]

    return codex_input


def codex_dir_checks(kwargs):
    try:
        # requd
        new_dirname = kwargs["name"]
    except:
        raise KeyError(
            "In creating a new CODEX directory; requires <name of CODEX directory>."
        )
    try:
        # not required
        new_parent_dirname = kwargs["parent_dir"]
    except KeyError:
        new_parent_dirname = os.path.dirname(os.path.realpath("."))
        print(
            f"Field <parent_dir> was unspecified. Creating new CODEX directory parent directory, {new_parent_dirname}."
        )

    new_parent_dirname = os.path.realpath(new_parent_dirname)

    try:
        assertpath = os.path.realpath(os.path.join(os.getcwd(), new_parent_dirname))
        assert os.path.exists(assertpath)
    except AssertionError:
        raise FileNotFoundError(
            f"Creating template CODEX directory failed. Folder {assertpath} does not exist."
        )

    try:
        include_templates = str.lower(kwargs["include_templates"]) == "true"
    except KeyError:
        include_templates = False
    try:
        include_tutorial = str.lower(kwargs["include_examples"]) == "true"
    except KeyError:
        include_tutorial = False

    return new_dirname, new_parent_dirname, include_templates, include_tutorial
