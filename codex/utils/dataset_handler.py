import os
import pandas as pd


def reorder_df_byindex(df_to_sort, order_ascending=True):
    if type(df_to_sort) is dict:
        print("Sorting and transposing performance JSON...")
        df_to_sort = pd.DataFrame(df_to_sort).sort_index().transpose()
    assert type(df_to_sort) is pd.DataFrame
    df_to_sort = df_to_sort.sort_index(ascending=order_ascending)
    return df_to_sort


def reorder_df_by_sample(
    sample_id_col: str, df_to_sort: pd.DataFrame, order_ascending=True
):
    df_to_sort = df_to_sort.sort_values(by=[sample_id_col], ascending=order_ascending)
    df_to_sort = df_to_sort.reset_index().drop(columns="index")
    return df_to_sort


def df_slice_by_id_reorder(
    sample_id_col: str, dataset_df: pd.DataFrame, sample_ids: list, order_ascending=True
):
    sliced_df = dataset_df.loc[dataset_df[sample_id_col].isin(sample_ids)]
    sliced_df = (
        sliced_df.sort_values(by=sample_id_col, ascending=order_ascending)
        .reset_index()
        .drop(columns="index")
    )
    return sliced_df


"""trainDF = dataset_df.loc[dataset_df[input['sample_id_column']].isin(train_ids)]
    trainDF = trainDF.sort_values(by=[input['sample_id_column']], ascending=True).reset_index().drop(columns='index')
    test_ids = split['test{}'.format(test_subset_tag)]
    testDF = dataset_df.loc[dataset_df[input['sample_id_column']].isin(test_ids)] #.reset_index().drop(columns='index')
    testDF = testDF.sort_values(by=[input['sample_id_column']], ascending=True).reset_index().drop(columns='index')
    for id in list(train_df_sorted[sample_id_col]):
        assert id not in list(test_df_sorted[sample_id_col])"""
