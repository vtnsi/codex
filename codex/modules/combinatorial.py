# Author: Erin Lanus lanus@vt.edu
# Updated Date: 4 April 2024
# Movement to new repo: 12 Apr, 2024

# Note: uncomment #plt.show()

# -------------------- Imported Modules --------------------
import os
import math
import random
import numpy as np
import pandas as pd
import json
from scipy.special import comb
import copy
import logging

from ..modules import output
#from codex.modules import output

# -------------------- Global Variables               --------------------#
labelCentric = False
verbose = False
id = None
identifyImages = False

LOGGER_COMBI = logging.getLogger(__name__)

# --- Functions for Encoding/Decoding To/From Combinatorial Abstraction ----#


# object that represents a mapping between the problem instance as tab delimited text file of metadata and labels
# and the abstraction over which the combinatorial code runs
# features is a list of the column headers (metadata features), indexed by column order
# values is a list of lists, one list per column indexed by column order, sublists are values that feature can have
# indexed by order of appearance. Assumes binning already done.
# data is list of lists, one list per non-column header row of the input file in numerical values (abstraction)
class Mapping:
    def __init__(self, features, values, data):
        self.features = features
        self.values = values
        self.data = data

    def __repr__(self):
        representation = (
            str(self.features) + ", " + str(self.values) + ", " + str(self.data)
        )
        return representation


# takes in a dataframe DF, maybe a map
# if a mapping provided as argument, creates a binding the features and values, appends to the values any new values
# produces the mapping of actual values in the problem instance to abstraction (numerical values)
# stores as an object of the Mapping class
# multiple file versions assume that both files have exactly the same features and in the same order
# todo add check code to break assumption, matches on feature name instead of placement
def encoding(DF, mMap, createData):
    if mMap != None:
        features = (
            mMap.features
        )  # create a binding from input features to this file's features
        values = (
            mMap.values
        )  # create a binding from input values to this file's features
    else:
        # save the column headers as a list
        features = copy.deepcopy(DF.schema.names)
        # don't use id column as a feature
        if features[0].find(id) > -1:
            features.pop(0)
        values = [[] for col in range(0, len(features))]
    # if createData set to True, produces the data structure as well
    # set to false to just create the feature and value mapping
    # initalize data to have the same number of rows as DF and same number of non-ID features
    data = None
    if createData:
        rows = len(DF.index)  # DF.count()
        cols = len(features)
        data = np.zeros(shape=(rows, cols), dtype=np.uint8, order="C")
        for col in range(0, cols):
            column = DF[features[col]]  # .topd()[features[col]]
            # now go through all of the rows for that column and replace with the mapped value
            for row in range(0, len(data)):
                # look up index for val and store the index in the data lists
                data[row][col] = values[col].index(column[row])
    return Mapping(features, values, data)


# takes a pair of column, value from abstraction and the map object
# returns metadata feature name and value name from map (maps back to original value)
def decoding(representation, col, val):
    return representation.features[col], representation.values[col][val]


def decodingCombo(representation, col, val):
    return representation.features[col]


# takes a rank, size, and an int array of that size and fills in the t set of columns
# corresponding to that rank in colexicographic order into the set


def rankToTuple(set, rank, size):
    if size == 0:  # base case
        return set
    m = size - 1
    while comb(m + 1, size, exact=True) <= rank:
        m += 1
    newRank = rank - comb(m, size, exact=True)
    # recursive call, returns set populated with later values
    set = rankToTuple(set, newRank, size - 1)
    # was set[size-1] = m, inserting each element before what was entered on last recursive loop
    set.append(m)
    return set


# converts a combination from a tuple to a value in mixed radix to use as an index into lists of counts
# this method chosen hopefully for speed improvements later; treating list as an array as it's an ordered data structure
# e.g. if t=3, cell values are c1 c2 c3, number of values for the columns are v1 v2 v3
# combinationIndex = c1*1 + c2*v1 + c3*v1*v2
# multiplier keeps tacking on the last level seen
def convertValueComboToIndex(representation, set, row):
    index = 0
    multiplier = 1
    for col in range(0, len(set)):
        index += representation.data[row][set[col]] * multiplier
        #print(representation.values[set[col]])
        multiplier *= len(representation.values[set[col]])
    return index


# takes in an index corresponding to an interaction and produces the set of symbols
# need to know the rank and the index to produce the values
# index is a mixed radix value of the symbols
# column tuple is the set of t columns computed from the rank
# this function is doing the reverse of convertValueComboToIndex
# e.g. if t=3, cell values are c1 c2 c3, number of values for the columns are v1 v2 v3
# combinationIndex = c1*1 + c2*v1 + c3*v1*v2
# determine the divisor for the last position of the tset then works backwards
def convertIndexToValueCombo(representation, index, set, t):
    values = [0 for index in range(0, t)]
    divisor = 1
    for pos in range(0, t):
        # set[pos] returns the column in position pos of the set
        # len(representation.values[set[pos]]) gives the number of values for that column
        divisor *= len(representation.values[set[pos]])
    # starting at the last position, t-1, and moving to position 0 of the tset
    # update the divisor (done first instead of last so not to divide by invalid memory)
    # compute the quotient (symbol at that position) and remainder (value to carry, stored in index)
    for pos in range(t - 1, -1, -1):
        divisor /= len(representation.values[set[pos]])
        # integer division gets the quotient
        values[pos] = int(index // divisor)
        index = index % divisor  # gets the remainder
    return values


# -------------------- Functions Compute Coverage     -------------------- #


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
    if labelCentric:  # if labelCentric,
        tprime -= 1  # require t-1 way interactions of all other columns
        kprime -= 1  # and only consider the first k-1 columns in the rank
    # k choose t - number of  combinations of t columns
    kct = comb(kprime, tprime, exact=True)

    LOGGER_COMBI.log(
        level=25,
        msg="k: {}\n k': {}\n, t: {}\n, t': {}\n, k choose t: {}".format(k, kprime, t, tprime, kct)
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
        set = rankToTuple(set, rank, tprime)
        interactionsForRank = 1  # compute v1*v2*...*vt so start with 1
        if labelCentric:  # if every combination must include the label column
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
            index = convertValueComboToIndex(representation, set, row)
            symbols = []

            # NOTE, 06.06.24: this loop is huge
            # symbols = convertIndexToValueCombo(representation, index, set, t)
            # logging.getLogger(__name__).debug('Row \t{}\t has symbols: {} corresponding to index {}'.format(row, symbols, index))
            countsEachCombination[index] += 1

        countsAllCombinations.append(countsEachCombination)
        LOGGER_COMBI.log(
            level = 15,
            msg = "Rank:{}, set:{}, combinations:{}".format(rank, set, interactionsForRank)
        )
        LOGGER_COMBI.log(
            level = 15,
            msg = "Counts of each combination:{}".format(countsEachCombination)
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
        LOGGER_COMBI.log(level = 15,
                         msg = targetCoverage['countsAllCombinations'][rank])

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

    LOGGER_COMBI.log(level=15,
                     msg=setDifferenceStructure)
    
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
        LOGGER_COMBI.log(level=15,
                         msg = targetCoverage['countsAllCombinations'][rank])

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
                    counts[interaction] = -1 # REVERTED FROM 2 05/16/25
                else:
                    assert counts[interaction] == -1
        setDifferenceStructure["setDifferenceInteractionsCounts"].append(counts)
    setDifferenceStructure["interactionsInTarget"] = interactionsInTarget
    setDifferenceStructure["setDifferenceInteractions"] = setDifferenceInteractions

    LOGGER_COMBI.log(level=15,
                     msg = 'Set difference structure: {}'.format(setDifferenceStructure))

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
    if labelCentric:
        tprime -= 1  # require t-1 way interactions of all other columns

    LOGGER_COMBI.log(msg='t: {}, t\': {}, k choose t: {}'.format(t, tprime, kct),
                     level=25)

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
                tupleCols = rankToTuple(set, rank, tprime)
                # the rank is for the t-1 cols not including the label col
                if labelCentric:
                    set.append(k - 1)
                tupleValues = convertIndexToValueCombo(representation, index, set, t)
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

    LOGGER_COMBI.log(msg='Missing interactions: {}'.format(missing),
                     level=25)

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
    if labelCentric:
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
                tupleCols = rankToTuple(set, rank, tprime)
                # the rank is for the t-1 cols not including the label col
                if labelCentric:
                    set.append(k - 1)
                tupleValues = convertIndexToValueCombo(representation, index, set, t)
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


# decode the interactions list into original representation (not abstraction)
def decodeInteraction(representation, interaction):
    decodedInteraction = []
    for pair in interaction:
        decodedInteraction.append(decoding(representation, *(pair)))
    return decodedInteraction


# decode the missing interactions list into original representation (not abstraction)


def decodeMissingInteractions(representation, missing):
    decodedMissing = []
    for interaction in missing:
        decodedInteraction = []
        for pair in interaction:
            decodedInteraction.append(decoding(representation, *(pair)))
        decodedMissing.append(decodedInteraction)
    return decodedMissing


# decode the missing interactions list into original representation (not abstraction)


def decodeSetDifferenceInteractions(representation, setDifferenceInteractions):
    decodedSetDifference = []
    for interaction in setDifferenceInteractions:
        decodedInteraction = []
        for pair in interaction:
            decodedInteraction.append(decoding(representation, *(pair)))
        decodedSetDifference.append(decodedInteraction)
    return decodedSetDifference


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
    if labelCentric:
        tprime -= 1  # require t-1 way interactions of all other columns
    # go rank by rank through kct ranks
    for rank in range(0, kct):
        numinteractionsForRank = len(CC["countsAllCombinations"][rank])
        countAppearingInteractionsInRank = 0
        for index in range(0, numinteractionsForRank):
            # unranks and returns the set
            set = []
            tupleCols = rankToTuple(set, rank, tprime)
            # the rank is for the t-1 cols not including the label col
            if labelCentric:
                set.append(k - 1)
            tupleValues = convertIndexToValueCombo(data, index, set, t)
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
                    decodedCombination += str(decodingCombo(data, *(pair)))
                else:
                    decodedCombination += "*{}".format(
                        str(decodingCombo(data, *(pair)))
                    )
                decodedInteraction += str(decoding(data, *(pair)))

        if decodedCombination not in ranks:
            ranks.append(decodedCombination)

    return ranks


# main entry point for computing CC on one file
# expects name for labeling on plots


def CC_main(dataDF, name, universe, t, output_dir):
    """
    Main entry point to computing combinatorial coverage over data.
    """

    global verbose, labelCentric, identifyImages
    labelCentric = False
    mMap = Mapping(universe["features"], universe["levels"], None)
    data = encoding(dataDF, mMap, True)

    LOGGER_COMBI.log(msg="Metadata level map:\n{}".format(mMap), level=15)
    LOGGER_COMBI.log(msg="Data representation:\n{}".format(data), level=15)

    k = len(data.features)
    if t > k:
        print("t =", t, " cannot be greater than number of features k =", k)
        return

    # computes CC for one dataset
    CC = combinatorialCoverage(data, t)

    LOGGER_COMBI.log(msg="CC:{}".format(CC), level=15)

    counts = CC["countsAllCombinations"]
    ranks = decodeCombinations(data, CC, t)

    decodedMissing = decodeMissingInteractions(
        data, computeMissingInteractions(data, CC)
    )
    # create t file with results for this t -- CC and missing interactions list
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, "CC")):
        os.makedirs(os.path.join(output_dir, "CC"))

    ranks = decodeCombinations(data, CC, t)
    output.writeCCtToFile(output_dir, name, t, CC)
    output.writeMissingtoFile(output_dir, name, t, decodedMissing)

    jsondict = cc_dict(CC)
    jsondict["combinations"] = ranks
    jsondict["combination counts"] = counts
    jsondict["missing interactions"] = decodedMissing

    return jsondict


# main entry point for computing SDCC on two files
# expects sourceName for labeling on plots and sourceFile as file name of binned csv file
# and same for target


# (path, sourceName, sourceFile, targetName, targetFile, metadata, t):
def SDCC_main(
    sourceDF,
    sourceName,
    targetDF,
    targetName,
    universe,
    t,
    output_dir,
    comparison_mode,
    split_id=None,
):
    """
    Main entry point to computing set difference combinatorial coverage between two
    datasets. Set difference meaning, that which is appearing in a target dataset
    that does not exist in a source dataset.
    """
    global verbose, labelCentric, identifyImages

    mMap = Mapping(universe["features"], universe["levels"], None)
    source_data = encoding(sourceDF, mMap, True)
    target = encoding(targetDF, mMap, True)

    LOGGER_COMBI.log(msg="Metadata level map:\n{}".format(mMap), level=15)
    LOGGER_COMBI.log(msg="'Source' data representation:\n{}".format(source_data), level=15)
    LOGGER_COMBI.log(msg="'Target' data representation:\n{}".format(target), level=15)

    k = len(source_data.features)

    # no combos appearing
    sourceCC = combinatorialCoverage(source_data, t)
    targetCC = combinatorialCoverage(target, t)

    LOGGER_COMBI.log(msg="Source CC:\n".format(sourceCC), level=15)
    LOGGER_COMBI.log(msg="Target CC:\n".format(targetCC), level=15)


    """Caution: 06.03.24"""
    if comparison_mode:
        output_dir = output.make_output_dir_nonexist(os.path.join(output_dir, split_id))
    else:
        output_dir = output.make_output_dir_nonexist(output_dir)

    source_ranks = decodeCombinations(source_data, sourceCC, t)
    target_ranks = decodeCombinations(target, targetCC, t)
    assert source_ranks == target_ranks

    sourceDecodedMissing = decodeMissingInteractions(
        source_data, computeMissingInteractions(source_data, sourceCC)
    )
    targetDecodedMissing = decodeMissingInteractions(
        target, computeMissingInteractions(target, targetCC)
    )

    # create t file with results for this t -- CC and missing interactions list

    # create t file with results for this t -- CC and missing interactions list

    # compute set difference target \ source
    SDCCconstraints = setDifferenceCombinatorialCoverageConstraints(sourceCC, targetCC)
    output.writeSDCCtToFile(output_dir, sourceName, targetName, t, SDCCconstraints)
    setDifferenceInteractions = computeSetDifferenceInteractions(
        target, SDCCconstraints
    )
    decodedSetDifferenceInteractions = decodeSetDifferenceInteractions(
        target, setDifferenceInteractions
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
    reversedecodedSetDifferenceInteractions = decodeSetDifferenceInteractions(
        source_data, reversesetDifferenceInteractions
    )
    output.writeSetDifferencetoFile(
        output_dir, targetName, sourceName, t, reversedecodedSetDifferenceInteractions
    )

    # if identifyImages and len(decodedSetDifferenceInteractions) > 0:
    #            identifyImagesWithSetDifferenceInteractions(sourceDF, reversedecodedSetDifferenceInteractions)

    sdccdict = {}

    sdccdict[sourceName] = cc_dict(sourceCC)
    sdccdict[sourceName]["combinations"] = source_ranks
    sdccdict[sourceName]["combination counts"] = sourceCC["countsAllCombinations"]
    sdccdict[sourceName]["missing interactions"] = sourceDecodedMissing

    sdccdict[targetName] = cc_dict(targetCC)
    sdccdict[targetName]["missing interactions"] = targetDecodedMissing
    sdccdict[targetName]["combinations"] = target_ranks
    sdccdict[targetName]["combination counts"] = targetCC["countsAllCombinations"]
    sdccdict[targetName]["missing interactions"] = targetDecodedMissing

    sdccdict[targetName + "-" + sourceName] = sdcc_dict(SDCCconstraints)
    sdccdict[targetName + "-" + sourceName]["combinations"] = source_ranks
    sdccdict[targetName + "-" + sourceName]["sdcc counts"] = SDCCconstraints[
        "setDifferenceInteractionsCounts"
    ]

    sdccdict[sourceName + "-" + targetName] = sdcc_dict(reverseSDCCconstraints)
    sdccdict[sourceName + "-" + targetName]["combinations"] = source_ranks
    sdccdict[sourceName + "-" + targetName]["sdcc counts"] = reverseSDCCconstraints[
        "setDifferenceInteractionsCounts"
    ]

    return sdccdict


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
    if labelCentric:
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
            tupleCols = rankToTuple(set, rank, tprime)
            # the rank is for the t-1 cols not including the label col
            if labelCentric:
                set.append(k - 1)
            tupleValues = convertIndexToValueCombo(data, index, set, CC["t"])
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

            LOGGER_COMBI.log(msg = "Rank {}, index {} appears {} times, frequency of {}.".format(str(rank), str(index), 
                                                                                                 appearancesThisInteraction, 
                                                                                                 round(percentageOfAllInteractions, 3), 
                                                                                                 decodeInteraction(data, interaction)),
                                                                                                 level=15)
                

            if appearancesThisInteraction < goalSamples[rank]:
                print(
                    decodeInteraction(data, interaction),
                    "INSUFFICIENT SAMPLES ",
                    appearancesThisInteraction,
                    " TO MEET TEST SET GOAL",
                    goalSamples[rank],
                )
            elif appearancesThisInteraction < (goalSamples[rank] * 2):
                print(
                    decodeInteraction(data, interaction),
                    "SAMPLE COUNT ",
                    appearancesThisInteraction,
                    " < 2x TEST SET GOAL",
                    goalSamples[rank],
                )

        LOGGER_COMBI.log(msg = "Coverage for rank {} is {}".format(
                rank, countAppearingInteractionsInRank / numInteractionsForRank
            ),
            level=25)
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
    if labelCentric:  # if labelCentric,
        tprime -= 1  # require t-1 way interactions of all other columns
        kprime -= 1  # and only consider the first k-1 columns in the rank
    kct = CC["kct"]
    for rank in range(0, kct):  # go rank by rank
        # enumerate all combinations
        set = []  # start with an empty set to pass through to the function
        # find the set of columns corresponding to this rank
        set = rankToTuple(set, rank, tprime)
        interactionsForRank = 1  # compute v1*v2*...*vt so start with 1
        if labelCentric:  # if every combination must include the label column
            set.append(kprime)  # add the last column (label) to the set
        # it's t not t' as every combination is of t columns regardless
        for column in range(0, t):
            interactionsForRank *= len(representation.values[set[column]])
        if interactionsForRank != len(CC["countsAllCombinations"][rank]):
            print("interactionsForRank error")
            exit(-1)
        index = convertValueComboToIndex(representation, set, row)
        # if verbose:
        #    symbols = convertIndexToValueCombo(representation, index, set, t)
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
    if labelCentric:  # if labelCentric,
        tprime -= 1  # require t-1 way interactions of all other columns
        kprime -= 1  # and only consider the first k-1 columns in the rank
    kct = CC["kct"]

    for interaction in frequency:
        rank = interaction[0]
        index = interaction[1]
        f = interaction[3]
        set = []  # start with an empty set to pass through to the function
        # find the set of columns corresponding to this rank
        set = rankToTuple(set, rank, tprime)
        symbols = convertIndexToValueCombo(representation, index, set, t)
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


def findSamples(IDs, testCC, goal, sortedFrequency):
    t = testCC["t"]
    representation = testCC["representation"]
    kprime = testCC["k"]
    tprime = t
    if labelCentric:  # if labelCentric,
        tprime -= 1  # require t-1 way interactions of all other columns
        kprime -= 1  # and only consider the first k-1 columns in the rank
    kct = testCC["kct"]
    for interaction in sortedFrequency:
        rank = interaction[0]
        index = interaction[1]
        set = []  # start with an empty set to pass through to the function
        # find the set of columns corresponding to this rank
        set = rankToTuple(set, rank, tprime)
        symbols = convertIndexToValueCombo(representation, index, set, t)
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
            int(math.ceil(goal[rank])) - testCC["countsAllCombinations"][rank][index]
        )
        # print("numSamples: ", goal[rank], testCC['countsAllCombinations'][rank][index], numSamplesToSelect)
        if numSamplesToSelect > len(matchingIDs):
            raise AssertionError("Too few samples to meet goal")
            exit(-1)
        for i in range(0, numSamplesToSelect):
            modifyTest(IDs, testCC, matchingIDs[i][0], add=True)
            # print(matchingIDs[i][0], IDs[matchingIDs[i][0]])
    return


def testsetPostOptimization(IDs, testCC, goal):
    t = testCC["t"]
    representation = testCC["representation"]
    kprime = testCC["k"]
    tprime = t
    if labelCentric:  # if labelCentric,
        tprime -= 1  # require t-1 way interactions of all other columns
        kprime -= 1  # and only consider the first k-1 columns in the rank
    kct = testCC["kct"]
    redundancy = []
    for rank in range(0, kct):  # For each rank
        r = [rank, 0]  # rank: int, r: list, i: list
        i = []
        # For each interaction
        for interaction in range(0, len(testCC["countsAllCombinations"][rank])):
            # If number of interactions/goal for this combo > 0?
            if testCC["countsAllCombinations"][rank][interaction] / goal[rank] > r[1]:
                # Update [1] to be that proportion. Stores where there might be imbalance?
                r[1] = testCC["countsAllCombinations"][rank][interaction] / goal[rank]
            # print(goal[rank], testCC['countsAllCombinations'][rank][interaction], r)
            # In any case,  add interaction, number of appearences
            i.append([interaction, testCC["countsAllCombinations"][rank][interaction]])
        r.append(i)
        redundancy.append(r)

    LOGGER_COMBI.log(msg = "Interaction redundancy list:\n{}{}".format(redundancy, type(redundancy)),
                     level=15)
    redundancy.sort(key=lambda x: x[1], reverse=True)
    LOGGER_COMBI.log(msg = "Interaction redundancy list, sorted:\n{}{}".format(redundancy, type(redundancy)),
                     level=15)


    for r in redundancy:
        rank = r[0]
        interactions = r[2]
        interactions.sort(key=lambda x: x[1], reverse=True)

        LOGGER_COMBI.log(msg="Sorted list for finding redundant coverage:\nRank: {}, Interactions: {}".format(rank, interactions),
                         level=15)

        for i in interactions:
            index = i[0]
            set = []  # start with an empty set to pass through to the function
            # find the set of columns corresponding to this rank
            set = rankToTuple(set, rank, tprime)
            symbols = convertIndexToValueCombo(representation, index, set, t)
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
                        LOGGER_COMBI.log(level=15,
                                         msg = "Rank {}-index {} match on row {}".format(rank, index, row))

                        # determine if removing this row would drop any interaction below the goal for interaction coverage in test
                        remove = True
                        for rank2 in range(0, kct):  # go rank by rank
                            set2 = []  # start with an empty set to pass through to the function
                            # find the set of columns corresponding to this rank
                            set2 = rankToTuple(set2, rank2, tprime)
                            index2 = convertValueComboToIndex(representation, set2, row)

                            #    symbols = convertIndexToValueCombo(representation, index2, set2, t)
                            LOGGER_COMBI.log(level=15,
                                             msg = "Row \t{}\t has symbols: {} corresponding to index {}".format(
                                                row, symbols, index2
                                        )
                                    )
                            LOGGER_COMBI.log(level=15,
                                             msg="Row \t{}\t has symbols: {} corresponding to index {}".format(row, symbols, index2)
                                )

                            if (
                                testCC["countsAllCombinations"][rank2][index2]
                                <= goal[rank2]
                            ):
                                remove = False
                                break
                        if remove:
                            modifyTest(IDs, testCC, row, add=False)

                            LOGGER_COMBI.log(level=15,
                                             msg="Removing row {}, {}".format(
                                                row, representation.data[row]
                                            )
                                        )
                            LOGGER_COMBI.log(level=15,
                                             msg="Resultant test CC: {}".format(testCC)
                                        )
    return


# TODO: rename to SIE and split out functionality that is just balanced test set construction vs. entire SIE (train/val)
def balanced_test_set(
    dataDF,
    name,
    sampleID,
    universe,
    strengths,
    testSetSizeGoal,
    output_dir,
    include_baseline=True,
    baseline_seed=1,
    form_exclusions=None
):
    global verbose, labelCentric, identifyImages
    labelCentric = False
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, "exclusions"))
        os.makedirs(os.path.join(output_dir, "SIE_splits"))

    mMap = Mapping(universe["features"], universe["levels"], None)
    # produce a random ordering of the data prior to representation creation so multiple runs produce different candidate results
    dataDF = dataDF.sample(frac=1).reset_index(drop=True)
    data_representation = encoding(dataDF, mMap, True)

    LOGGER_COMBI.log(msg="Metadata level map:\n{}".format(mMap), level=15)
    LOGGER_COMBI.log(msg="Data representation:\n{}".format(data_representation), level=15)
    LOGGER_COMBI.log(msg="DataFrame:\n{}".format(dataDF), level=15)

    k = len(data_representation.features)
    t = max(strengths) # maximum t
    if t > k:
        print("t =", t, " cannot be greater than number of features k =", k)
        return

    # computes CC for one dataset
    CC = combinatorialCoverage(data_representation, t)
    LOGGER_COMBI.log(level=25, msg="CC: {}".format(CC))

    decodedMissing = decodeMissingInteractions(
        data_representation, computeMissingInteractions(data_representation, CC)
    )
    output.writeCCtToFile(output_dir, name, t, CC)
    output.writeMissingtoFile(output_dir, name, t, decodedMissing)

    # build test set in 2 phases
    # phase 1: add samples to test set with first priority to achieve coverage
    # and second priority to have balance
    # sort the interactions by rareness and go through them one at a time
    # for each interaction, select samples to cover the interaction at the
    # desired target and update coverage structure
    # to prioritize balance, score each sample by rareness of interactions contained
    # phase 2: post-optimization
    # determine if a sample can be removed due to redundancy (over coverage)

    testCC = copy.deepcopy(CC)
    for combinations in testCC["countsAllCombinations"]:
        for interaction in range(0, len(combinations)):
            combinations[interaction] = 0
    testCC["countAppearingInteractions"] = 0

    goal = goalSamples(CC, testSetSizeGoal)
    print("\nGoal # samples per rank in test set:\n", goal, "\n")
    frequency = frequencyInteractions(CC, goal)
    # print("\nFrequency of interactions: ", frequency)
    sortedFrequency = copy.deepcopy(frequency)
    sortedFrequency.sort(key=lambda x: x[3])

    LOGGER_COMBI.log(level=25, msg="\nGoal # samples per rank in test set: {}\n".format(goal))
    LOGGER_COMBI.log(level=25, msg="Sorted number and frequency of interactions: {}".format(sortedFrequency))

    # create a data structure that maps sample IDs from data to the index in the data representation of encoded
    # features of the sample as well as whether the sample has been added to the Test set
    # and give the sample a score that is the sum of interaction frequencies contained in the sample
    idlist = list(dataDF[sampleID])
    IDs = dict()  # Empty dict
    for index in range(0, len(data_representation.data)):
        IDs[index] = {"id": idlist[index], "inTest": False, "score": 0}

    # Pass 1 build the test set
    computeScoreIDs(IDs, CC, frequency)
    findSamples(IDs, testCC, goal, sortedFrequency)
    print("\nINITIAL TESTCC\n", testCC)
    # print("TEST", json.dumps(CC, indent=2))
    # Pass 2 reduce the test set
    testsetPostOptimization(IDs, testCC, goal)
    # print("TEST", json.dumps(IDs, indent=2))

    # determine the max training set size m as the minimum over all withheld interactions
    allsamplesize = sum(CC["countsAllCombinations"][0])
    m = allsamplesize
    testsamplesize = sum(testCC["countsAllCombinations"][0])
    kct = len(CC["countsAllCombinations"])
    print("\nRank, index, potential training sizes:")
    for rank in range(0, kct):  # go rank by rank
        for i in range(0, len(testCC["countsAllCombinations"][rank])):
            samplesnotintest = (
                CC["countsAllCombinations"][rank][i]
                - testCC["countsAllCombinations"][rank][i]
            )
            maxPoolForInteraction = allsamplesize - testsamplesize - samplesnotintest

            LOGGER_COMBI.log(level=25, msg="Rank {}, i {}, maximum pool size for interaction: {}".format(
                    rank, i, maxPoolForInteraction
                )
            )

            m = min(m, maxPoolForInteraction)
            print('!!!', m)

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
    querystring = sampleID + " in @trainpool"
    trainpoolDF = dataDF.query(querystring).reset_index(drop=True)

    """trainpoolDFrepresentation = encoding(trainpoolDF, mMap, True)
    trainpoolCC = combinatorialCoverage(trainpoolDFrepresentation, t)
    if trainpoolCC['totalPossibleInteractions'] !=trainpoolCC['countAppearingInteractions']:
        print('training pool does not contain some interaction. Exiting.')
        print(trainpoolCC)
        exit()"""

    # get all of the samples in the test set for later splitting into included/excluded subsets
    querystring = sampleID + " in @test"
    testsetDF = dataDF.query(querystring)
    testsetDF.to_csv(os.path.join(output_dir, "test.csv"))
    trainpoolDF.to_csv(os.path.join(output_dir, "trainpool.csv"))

    LOGGER_COMBI.log(level=15, msg="TRAINING POOL DF:\n{}".format(trainpoolDF))
    LOGGER_COMBI.log(level=15, msg="UNIVERSAL TEST SET DF:\n{}".format(testsetDF))

    jsondict = cc_dict(CC)
    jsondict["max_training_pool_size"] = m
    jsondict["universal_test_set_size"] = testsamplesize

    # Moved 12.10.24
    jsondict["max_t"] = t
    jsondict["countAllCombinations_data"] = CC["countsAllCombinations"]
    jsondict["countAllCombinations_test"] = testCC["countsAllCombinations"]

    if not form_exclusions:
        return jsondict

    # TODO: up until now is test creation, from here is remainder

    # TODO: update from feature to interaction, iterate through ranks
    # for each interaction in each combination make the list of train values
    # these are samples that are not in the test set and do not have the withheld interaction
    # of the remaining options, randomly select m of them
    # in the future, select m that balance the rest of the interactions as best as possible
    for f in range(len(data_representation.features)):
        feature = data_representation.features[f]
        for v in range(len(data_representation.values[f])):
            value = str(data_representation.values[f][v])
            modelnumber = str(f) + "_" + str(v) + "_"
            print("")
            print(data_representation.features[f] + " excluding " + value)
            # print("Trainpool excluding", value, "\n", trainpoolDF.query(representation.features[f] + ' not in @value'))

            # future algorithms should build intelligently instead of relying on random chance for coverage
            # but for now, if it's possible, resample until we have coverage (could be a very long time)
            # impossibility check - if the training pool does not cover all interactions aside from the withheld one
            # then it doesn't matter how many times we resample, the sampled training set will not cover all
            coveredpossible = True
            # trainpoolmodelDF = trainpoolDF.query(representation.features[f] + ' not in @value').reset_index(drop=True)
            trainpoolmodelDF = trainpoolDF[trainpoolDF[feature] != value].reset_index(
                drop=True
            )  # check if equivalent
            print('Trainpool dim', trainpoolmodelDF.shape)
            trainpoolmodelDFrepresentation = encoding(trainpoolmodelDF, mMap, True)
            trainpoolmodelCC = combinatorialCoverage(trainpoolmodelDFrepresentation, t)

            LOGGER_COMBI.log(level=15, msg="Training pool model counts: {}".format(
                    trainpoolmodelCC["countsAllCombinations"]
                    )
                )

            # TODO: update from feature to interaction
            for rank in range(0, kct):  # go rank by rank
                for i in range(0, len(testCC["countsAllCombinations"][rank])):
                    if (
                        rank != f
                        and i != v
                        and trainpoolmodelCC["countsAllCombinations"][rank][i] == 0
                    ):
                        coveredimpossible = (
                            True  # found a 0 that isn't the withheld interaction
                        )
                        m = len(trainpoolmodelDF) # EXP ADDED 12.03.2024
                        print(
                            "Warning: Model's Training Pool does not cover all other interactions, so constructed train can't either."
                        )

            while True:  # execute at least once no matter what
                # sample from whole train pool
                trainDF = trainpoolmodelDF.sample(m).reset_index(drop=True)
                
                LOGGER_COMBI.log(level=15, msg = "Number of samples containing {} in training set: {}".format(
                        value, len(trainDF[trainDF[feature] == value])
                    )
                )
                LOGGER_COMBI.log(level=25, msg="Training set:\n{}".format(trainDF.to_string()))

                # check that constructed training set covers everything EXCEPT the withheld interaction
                trainDFrepresentation = encoding(trainDF, mMap, True)
                trainCC = combinatorialCoverage(trainDFrepresentation, t)

                LOGGER_COMBI.log(level=25, msg="Train CC counts {}".format(trainCC["countsAllCombinations"]))

                # TODO: update from feature to interaction
                covered = True  # REFERNCING AFTER ASSIGNMENT BC UNBOUND LOCAL ERROR THIS MIGHT CAUSE ERRORS
                for rank in range(0, kct):  # go rank by rank
                    for i in range(0, len(testCC["countsAllCombinations"][rank])):
                        if (
                            rank != f
                            and i != v
                            and trainCC["countsAllCombinations"][rank][i] == 0
                        ):
                            covered = False
                            print(
                                "Constructed training set does not cover all other interactions. Will sample again."
                            )
                if covered or coveredimpossible:
                    break

            if not os.path.exists(os.path.join(output_dir, "splits_by_csv")):
                os.makedirs(os.path.join(output_dir, "splits_by_csv"))

            trainDF.to_csv(
                os.path.join(
                    output_dir, "splits_by_csv", "train_" + modelnumber + ".csv"
                )
            )

            LOGGER_COMBI.log(level=15, msg="\n Model {} excludes interaction {}, {}".format(
                    modelnumber, feature, value
                )
            )

            trainlist = list(trainDF[sampleID])
            cut = math.ceil(m * 0.8)
            # excludeTestDF = testsetDF.query(representation.features[f] + ' in @value')
            excludeTestDF = testsetDF[testsetDF[feature] == value]
            # includeTestDF = testsetDF.query(representation.features[f] + ' not in @value')
            includeTestDF = testsetDF[testsetDF[feature] != value]
            excludeTestDF.to_csv(
                os.path.join(
                    output_dir, "splits_by_csv", "notcovered_" + modelnumber + "_t{}.csv".format(t)
                )
            )
            includeTestDF.to_csv(
                os.path.join(
                    output_dir, "splits_by_csv", "covered_" + modelnumber + "_t{}.csv".format(t)
                )
            )

            includeTest = list(includeTestDF[sampleID])
            excludeTest = list(excludeTestDF[sampleID])
            jsondict["model_" + modelnumber] = {
                "test_included": includeTest,
                "test_excluded": excludeTest,
                "train": trainlist[:cut],
                "validation": trainlist[cut:],
            }

    if include_baseline:
        cut = math.ceil(m * 0.8)
        trainDF = trainpoolDF.sample(m)  # , random_state=baseline_seed)

        trainlist = list(trainDF[sampleID])
        includeTest = list(testsetDF[sampleID])

        jsondict["model_x_"] = {
            "test_included": includeTest,
            "test_excluded": [],
            "train": trainlist[:cut],
            "validation": trainlist[cut:],
        }
    
    jsondict["test"] = test
    jsondict["train_pool"] = trainpool

    return jsondict

    # -------------------- Compute Performance By Interaction --------------------#


# Requires a representation to be completed and a performance dictionary where
# name of key is performance metric and value is a list where each index is the performance for a sample
# it must have the same index as the representation so we can go row by row
# Time increase to compute coverage and performance by interaction simultaneously is minimal, just an extra index and increment per interaction
# Doing these sequentially requires 2 passes through the representation, so N * number of interactions
# Storage may be a factor
# Future versions could decouple these activities in an alternative function


def computePerformanceByInteraction(representation, t, performanceDF, test=True):
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
    if labelCentric:  # if labelCentric,
        tprime -= 1  # require t-1 way interactions of all other columns
        kprime -= 1  # and only consider the first k-1 columns in the rank
    # k choose t - number of  combinations of t columns
    kct = comb(kprime, tprime, exact=True)
    metrics = performanceDF.columns

    LOGGER_COMBI.log(level=15, msg="PERFORMANCE DF:\n{}".format(performanceDF))
    performanceDataStructure = {
        metric: [0] * kct for metric in metrics
    }  # dict.fromkeys(metrics)
    # for metric in metrics:
    #    performanceDataStructure[metric] =

    LOGGER_COMBI.log(level=25, msg="k: {}, k': {}, t: {}, t': {}, k choose t: {}".format(k, kprime, t, tprime, kct))
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
        set = rankToTuple(set, rank, tprime)
        interactionsForRank = 1  # compute v1*v2*...*vt so start with 1
        if labelCentric:  # if every combination must include the label column
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
            index = convertValueComboToIndex(representation, set, row)
            # if verbose:
            #    print("rank ", rank, " set ", set, " row ", row, " index ", index)
            # here we update the performance for the interaction by multiplying by the current count to get current performance
            # adding performance of the row then dividing by the new count
            currentCount = countsEachCombination[index]
            for metric in metrics:
                rowPerformance = performanceDF[metric].iloc[row]
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

    LOGGER_COMBI.log(level=25, msg="PERFORMANCE Data Structure:\n{}".format(performanceDF))

    return coverageDataStructure, performanceDataStructure


def decodePerformance(data, cc, perf):
    t = cc["t"]
    k = cc["k"]
    # either true k choose t or k' choose t' when label centric
    # previously computed correctly to store the combination counts so take list length
    kct = len(cc["countsAllCombinations"])
    humanReadablePerformance = {
        key for key in list(perf.keys())
    }  # dict.fromkeys(perf.keys(), {})
    tprime = t
    if labelCentric:
        tprime -= 1  # require t-1 way interactions of all other columns
    # go rank by rank through kct ranks
    for rank in range(0, kct):
        numinteractionsForRank = len(cc["countsAllCombinations"][rank])
        countAppearingInteractionsInRank = 0
        # for each combination in a rank
        for index in range(0, numinteractionsForRank):
            # unranks and returns the set
            set = []
            tupleCols = rankToTuple(set, rank, tprime)
            # the rank is for the t-1 cols not including the label col
            if labelCentric:
                set.append(k - 1)
            tupleValues = convertIndexToValueCombo(data, index, set, t)
            interaction = []
            for element in range(0, t):
                # pair is (col, val)
                pair = (tupleCols[element], tupleValues[element])
                # build the t-way interaction of pairs
                interaction.append(pair)
            decodedInteraction = ""
            for pair in interaction:
                decodedInteraction += str(decoding(data, *(pair)))
            for metric in humanReadablePerformance:
                performanceAtMetric = perf[metric][rank][index]
                humanReadablePerformance[metric][decodedInteraction] = (
                    performanceAtMetric
                )
    print(humanReadablePerformance)
    return humanReadablePerformance


def json_prettify(json_obj, file_path=None, sort=False, print_json=False):
    json_str = json.dumps(json_obj, sort_keys=sort, indent=4, separators=(",", ": "))
    if print_json:
        print(json_str)
        return
    with open(file_path, "w") as f:
        f.write(json_str)

    return json_str


def decodePerformanceGroupedCombination(data, cc, perf):
    t = cc["t"]
    k = cc["k"]
    # either true k choose t or k' choose t' when label centric
    # previously computed correctly to store the combination counts so take list length
    kct = len(cc["countsAllCombinations"])

    humanReadableCombinations = []
    humanReadablePerformance = {
        key: {} for key in list(perf.keys())
    }  # dict.fromkeys(perf.keys(),{})
    tprime = t
    if labelCentric:
        tprime -= 1  # require t-1 way interactions of all other columns
    # go rank by rank through kct ranks
    for rank in range(0, kct):
        for metric in humanReadablePerformance.keys():
            humanReadablePerformance[metric][rank] = {}
        numinteractionsForRank = len(cc["countsAllCombinations"][rank])
        countAppearingInteractionsInRank = 0
        # for each combination in a rank
        for index in range(0, numinteractionsForRank):
            # unranks and returns the set
            set = []
            tupleCols = rankToTuple(set, rank, tprime)
            # the rank is for the t-1 cols not including the label col
            if labelCentric:
                set.append(k - 1)
            tupleValues = convertIndexToValueCombo(data, index, set, t)
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
                    decodedCombination += str(decodingCombo(data, *(pair)))
                else:
                    decodedCombination += "*{}".format(
                        str(decodingCombo(data, *(pair)))
                    )
                decodedInteraction += str(decoding(data, *(pair)))

            # print(decodedCombination)
            # print(decodedInteraction)
            if decodedCombination not in humanReadableCombinations:
                humanReadableCombinations.append(decodedCombination)

            for metric in humanReadablePerformance:
                performanceAtMetric = None
                # print(rank, index)
                performanceAtMetric = perf[metric][rank][index]
                # print(metric, performanceAtMetric)

                humanReadablePerformance[metric][rank][decodedInteraction] = (
                    performanceAtMetric
                )
                # print(humanReadablePerformance)

    return humanReadablePerformance


def performanceByInteraction_main(
    test_dataDF: pd.DataFrame,
    train_dataDF: pd.DataFrame,
    performanceDF: pd.DataFrame,
    name,
    universe,
    t,
    output_dir,
    metric,
    sample_id,
    coverage_subset,
):
    """
    Main entry point for computing performance by interaction on a test set and combinatorial
    coverage over a training set from data.
    """
    global verbose, labelCentric, identifyImages
    labelCentric = False
    mMap = Mapping(universe["features"], universe["levels"], None)

    data = encoding(test_dataDF, mMap, True)
    data_train = encoding(train_dataDF, mMap, True)

    LOGGER_COMBI.log(msg="Metadata level map:\n{}".format(mMap), level=15)
    LOGGER_COMBI.log(msg="Data representation:\n{}".format(data), level=15)

    if performanceDF.index.values.tolist() != test_dataDF[sample_id].tolist():
        raise KeyError(
            "IDs in performance file do not match IDs in test set of split file"
        )

    k = len(data.features)
    if t > k:
        raise ValueError(
            "t = {} cannot be greater than number of features k = {}".format(t, k)
        )

    # computes CC for one dataset as well as the performance
    CC_test, perf = computePerformanceByInteraction(data, t, performanceDF=performanceDF)
    CC_train = combinatorialCoverage(data_train, t)

    if coverage_subset=='train':
        CC = CC_train
    elif coverage_subset=='test':
        CC=CC_test
    else:
        raise KeyError('Coverage over subset {} not found in split file.'.format(coverage_subset))

    LOGGER_COMBI.log(level=15, msg="CC over train: {}".format(CC))

    decodedMissing = decodeMissingInteractions(
        data, computeMissingInteractions(data, CC)
    )
    # create t file with results for this t -- CC and missing interactions list

    output.writeCCtToFile(output_dir, name, t, CC)
    output.writeMissingtoFile(output_dir, name, t, decodedMissing)

    jsondict = cc_dict(CC)
    jsondict["performance"] = perf
    jsondict["human readable performance"] = decodePerformanceGroupedCombination(
        data, CC, perf
    )
    test_ranks = decodeCombinations(data_train, CC, t)
    ranks = decodeCombinations(data_train, CC, t)
    jsondict["missing interactions"] = decodedMissing
    jsondict["combinations"] = ranks
    jsondict["combination counts"] = CC["countsAllCombinations"]
    assert test_ranks == ranks

    jsondict["coverage subset"] = coverage_subset
    return jsondict

def performance_by_frequency_coverage_main(trainpool_df, test_df_balanced, entire_df_cont, universe, t, output_dir, skew_level, id=None):
    import utils.pbfc_data as data
    import utils.pbfc_biasing as biasing
    import utils.pbfc_ml as classifier

    EXP_NAME = id
    
    combination_list = biasing.get_combinations(universe, t)
    for combination in combination_list:
        indices_per_interaction, combination_names = biasing.interaction_indices_t2(df=trainpool_df)

        train_df_biased, train_df_selected_filename, combo_int_selected, interaction_selected = biasing.skew_dataset_relative(
            df=trainpool_df, 
            interaction_indices=indices_per_interaction, 
            skew_level=skew_level,
            extract_combination=combination,
            output_dir=output_dir)
        
        train_df = entire_df_cont.loc[entire_df_cont.index.isin(train_df_biased.index.tolist())]
        test_df = entire_df_cont.loc[entire_df_cont.index.isin(test_df_STATIC_1211.index.tolist())]

        full_df_combo = pd.concat([train_df, test_df], axis=0)
        full_df_cont_combo_filename = '{}-{}_Skewed.csv'.format(original_data_file_name, EXP_NAME)
        #full_df_combo.to_csv(os.path.join(data_dir_skew, full_df_cont_combo_filename))
    
        X_train, X_test, Y_train, Y_test, split_filename = data.prep_split_data(None, train_df=train_df, test_df=test_df, name=EXP_NAME, drop_list=drop_list,
                                                                                        split_dir=split_dir, target_col='Diabetes_binary', id_col='ID')

        perf_filenames = classifier.model_suite(X_train, Y_train, X_test, Y_test, 
                            experiment_name=EXP_NAME,
                            output_dir = performance_dir,
                            scaler=scaler)
        
        for perf_filename in perf_filenames:
            input_dict_new = copy.deepcopy(INPUT_DICT)

            if '_gnb' in perf_filename:
                model_name = 'Gaussian Naive Bayes'
                model_name_small = 'gnb'
            elif '_lr' in perf_filename:
                model_name = 'Logistic Regression'
                model_name_small = 'lr'
            elif '_rf' in perf_filename:
                model_name = 'Random Forest'
                model_name_small = 'rf'
            elif '_knn' in perf_filename:
                model_name = 'KNN'
                model_name_small = 'knn'
            elif '_svm' in perf_filename:
                model_name = 'SVM'
                model_name_small = 'svm'
            else:
                print("No model name found!")

            # Static
            input_dict_new['metric'] = metric
            input_dict_new['dataset_file'] = full_df_cont_combo_filename
            input_dict_new['split_file'] = split_filename
            # Change per model
            save_dir = '_runs/pbi_pipeline/pbi-{}-{}'.format(EXP_NAME, model_name_small)
            input_dict_new['config_id'] = save_dir
            input_dict_new['model_name'] = model_name
            input_dict_new['dataset_name'] = "CDC Diabetes, skewed {}, {}".format(skew_level, model_name)
            input_dict_new['performance_file'] = perf_filename
                
            # CODEX ~~~~~~~~~~~~~~~
                # of chosen combo, skew_level, model
            result = codex.run(input_dict_new, verbose='1')
            results_multiple_model[model_name_small] = {'coverage': result, 'save_dir': save_dir}
    
    print("CHECK DOESNT CHANGE:", interaction_selected)
    results_multiple_model['interaction_skewed'] = interaction_selected
    results_multiple_model['training_size'] = len(train_df_biased)

    return jsondict