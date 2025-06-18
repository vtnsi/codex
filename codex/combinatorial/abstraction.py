import numpy as np
import pandas as pd
import copy
from scipy.special import comb

label_centric = False

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


# takes in a dataframe dataset_df, maybe a map
# if a mapping provided as argument, creates a binding the features and values, appends to the values any new values
# produces the mapping of actual values in the problem instance to abstraction (numerical values)
# stores as an object of the Mapping class
# multiple file versions assume that both files have exactly the same features and in the same order
# todo add check code to break assumption, matches on feature name instead of placement
def encoding(dataset_df: pd.DataFrame, mMap, createData):
    if mMap != None:
        features = (
            mMap.features
        )  # create a binding from input features to this file's features
        values = (
            mMap.values
        )  # create a binding from input values to this file's features
    else:
        # save the column headers as a list
        features = copy.deepcopy(dataset_df.schema.names)
        # don't use id column as a feature
        if features[0].find(id) > -1:
            features.pop(0)
        values = [[] for col in range(0, len(features))]
    # if createData set to True, produces the data structure as well
    # set to false to just create the feature and value mapping
    # initalize data to have the same number of rows as dataset_df and same number of non-ID features
    data = None
    if createData:
        rows = len(dataset_df.index)  # dataset_df.count()
        cols = len(features)
        data = np.zeros(shape=(rows, cols), dtype=np.uint8, order="C")
        for col in range(0, cols):
            column = dataset_df[features[col]]  # .topd()[features[col]]
            # now go through all of the rows for that column and replace with the mapped value
            for row in range(0, len(data)):
                # look up index for val and store the index in the data lists
                # Deprecated June 12 2025: data[row][col] = values[col].index(column[row])
                data[row][col] = values[col].index(column.iloc[row])
    return Mapping(features, values, data)


# takes a pair of column, value from abstraction and the map object
# returns metadata feature name and value name from map (maps back to original value)
def decoding(representation, col, val):
    return representation.features[col], representation.values[col][val]


def decoding_combo(representation, col, val):
    return representation.features[col]


# takes a rank, size, and an int array of that size and fills in the t set of columns
# corresponding to that rank in colexicographic order into the set


def rank_to_tuple(set: set, rank: int, size: int):
    if size == 0:  # base case
        return set
    m = size - 1
    while comb(m + 1, size, exact=True) <= rank:
        m += 1
    newRank = rank - comb(m, size, exact=True)
    # recursive call, returns set populated with later values
    set = rank_to_tuple(set, newRank, size - 1)
    # was set[size-1] = m, inserting each element before what was entered on last recursive loop
    set.append(m)
    return set


# converts a combination from a tuple to a value in mixed radix to use as an index into lists of counts
# this method chosen hopefully for speed improvements later; treating list as an array as it's an ordered data structure
# e.g. if t=3, cell values are c1 c2 c3, number of values for the columns are v1 v2 v3
# combinationIndex = c1*1 + c2*v1 + c3*v1*v2
# multiplier keeps tacking on the last level seen
def convert_value_combo_to_index(representation, set, row):
    index = 0
    multiplier = 1
    for col in range(0, len(set)):
        index += representation.data[row][set[col]] * multiplier
        # print(representation.values[set[col]])
        multiplier *= len(representation.values[set[col]])
    return index


# takes in an index corresponding to an interaction and produces the set of symbols
# need to know the rank and the index to produce the values
# index is a mixed radix value of the symbols
# column tuple is the set of t columns computed from the rank
# this function is doing the reverse of convert_value_combo_to_index
# e.g. if t=3, cell values are c1 c2 c3, number of values for the columns are v1 v2 v3
# combinationIndex = c1*1 + c2*v1 + c3*v1*v2
# determine the divisor for the last position of the tset then works backwards
def convert_index_to_value_combo(representation, index, set, t):
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


# decode the interactions list into original representation (not abstraction)
def decode_interaction(representation, interaction):
    decodedInteraction = []
    for pair in interaction:
        decodedInteraction.append(decoding(representation, *(pair)))
    return decodedInteraction


# decode the missing interactions list into original representation (not abstraction)


def decode_missing_interactions(representation, missing):
    decodedMissing = []
    for interaction in missing:
        decodedInteraction = []
        for pair in interaction:
            decodedInteraction.append(decoding(representation, *(pair)))
        decodedMissing.append(decodedInteraction)
    return decodedMissing


# decode the missing interactions list into original representation (not abstraction)


def decode_set_difference_interactions(representation, setDifferenceInteractions):
    decodedSetDifference = []
    for interaction in setDifferenceInteractions:
        decodedInteraction = []
        for pair in interaction:
            decodedInteraction.append(decoding(representation, *(pair)))
        decodedSetDifference.append(decodedInteraction)
    return decodedSetDifference


def decode_performance(data, cc, perf):
    t = cc["t"]
    k = cc["k"]
    # either true k choose t or k' choose t' when label centric
    # previously computed correctly to store the combination counts so take list length
    kct = len(cc["countsAllCombinations"])
    humanReadablePerformance = {
        key for key in list(perf.keys())
    }  # dict.fromkeys(perf.keys(), {})
    tprime = t
    if label_centric:
        tprime -= 1  # require t-1 way interactions of all other columns
    # go rank by rank through kct ranks
    for rank in range(0, kct):
        numinteractionsForRank = len(cc["countsAllCombinations"][rank])
        countAppearingInteractionsInRank = 0
        # for each combination in a rank
        for index in range(0, numinteractionsForRank):
            # unranks and returns the set
            set = []
            tupleCols = rank_to_tuple(set, rank, tprime)
            # the rank is for the t-1 cols not including the label col
            if label_centric:
                set.append(k - 1)
            tupleValues = convert_index_to_value_combo(data, index, set, t)
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


def decode_performance_grouped_combination(data, cc, perf):
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
    if label_centric:
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
            tupleCols = rank_to_tuple(set, rank, tprime)
            # the rank is for the t-1 cols not including the label col
            if label_centric:
                set.append(k - 1)
            tupleValues = convert_index_to_value_combo(data, index, set, t)
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
                    decodedCombination += str(decoding_combo(data, *(pair)))
                else:
                    decodedCombination += "*{}".format(
                        str(decoding_combo(data, *(pair)))
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
