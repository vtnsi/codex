from utils import input_handler

def dse_prereq_check(codex_input):
    split, performance, metric = input_handler.extract_sp(codex_input)

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
    split, performance, metric = input_handler.extract_sp(codex_input)
    try:
        assert len(split) == len(performance)
    except:
        raise IndexError(
            "Ensure that each split file has their own corresponding performance file."
        )
