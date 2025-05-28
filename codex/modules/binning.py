# Author: Erin Lanus lanus@vt.edu
# Created Date: Feb 2 2023
# Updated Date: Mar 25 2024
# Movement to new repo: 12 Apr, 2024

import pandas
import sys

def binfile(unbinnedfile, binningfile, features_original):
    DF = pandas.read_csv(unbinnedfile)

    # collect the features and bin levels from the bin file
    universe = dict()
    universe["features"] = []
    universe["levels"] = []
    with open(binningfile) as binfile:
        for line in binfile:
            feature_levels_list = line.split(": ")
            feature = feature_levels_list[0]
            linelist = feature_levels_list[1].split("\n")[0].split(";")
            levels = []

            if feature not in features_original:
                continue

            for element in linelist:
                levels.append(element)
            universe["features"].append(feature)
            universe["levels"].append(levels)

    # go through the csv and replace each value with the bin
    na_dropped = None

    if DF.isnull().values.any():
        na_dropped = DF[~DF.index.isin(DF.dropna().index)]
        DF = DF.dropna()

    for f in range(0, len(universe["features"])):
        # only bin numeric features.
        # included so that binning file can also allow for categorial variables to be given a uniform ordering
        # not dependent on appearance in data file
        feature = universe["features"][f]
        # split the level string into components endpoint inclusion/exclusion and value range
        firstChar = universe["levels"][f][0][0]
        if firstChar == "[" or firstChar == "(":
            levels = []
            for level in universe["levels"][f]:
                comma = level.index(",")

                levelbounds = [
                    level[0],
                    float(level[1:comma]),
                    float(level[comma + 1 : -1]),
                    level[-1],
                ]
                levels.append(levelbounds)

            # go through the rows for this feature and find the correct level bin for the value present
            rows = DF[feature].tolist()
            for i in range(0, len(rows)):
                value = rows[i]
                # REQUIRES SPECIFICATION IN BINNING FILE
                if type(value) is str:
                    pre_bin = value.split(",")
                    try:
                        """upper = float(pre_bin[1])
                        lower = float(pre_bin[0])"""
                        if len(pre_bin) != 1:
                            upper = float(pre_bin[1])
                            lower = float(pre_bin[0])
                            average_value_for_binning = (upper + lower) / 2
                        else:
                            average_value_for_binning = float(pre_bin[0])
                    except ValueError:
                        raise ValueError(
                            "String encountered without potential bin in correct format!",
                            feature,
                            value,
                        )
                    value = average_value_for_binning

                """if value == np.nan:
                    print("NaN value encountered. Continuing")
                    continue"""
                # print(value, type(value))
                found = False
                for j in range(0, len(levels)):
                    # figure out which level matches
                    if (
                        (
                            levels[j][0] == "["
                            and levels[j][3] == "]"
                            and levels[j][1] <= value
                            and value <= levels[j][2]
                        )
                        or (
                            levels[j][0] == "["
                            and levels[j][3] == ")"
                            and levels[j][1] <= value
                            and value < levels[j][2]
                        )
                        or (
                            levels[j][0] == "("
                            and levels[j][3] == "]"
                            and levels[j][1] < value
                            and value <= levels[j][2]
                        )
                        or (
                            levels[j][0] == "("
                            and levels[j][3] == ")"
                            and levels[j][1] < value
                            and value < levels[j][2]
                        )
                    ):
                        found = True
                        rows[i] = universe["levels"][f][j]
                        break

                if not found:
                    raise ValueError(
                        "Error: bin not found for value {}. Feature {}, level {} ".format(
                            value, feature, level
                        )
                    )
            colindex = DF.columns.get_loc(feature)
            del DF[feature]
            DF.insert(colindex, feature, rows, allow_duplicates=False)
    binnedfile = unbinnedfile.split(".csv")[0] + "_Binned.csv"
    dropfile = unbinnedfile.split(".csv")[0] + "_nadropped.csv"
    if na_dropped is not None:
        na_dropped.to_csv(dropfile, index=False)
    DF.to_csv(binnedfile, index=False)

    return universe, binnedfile


if __name__ == "__main__":
    unbinnedfile = sys.argv[1]
    binningfile = sys.argv[2]
    universe, binnedfile = binfile(unbinnedfile, binningfile)
