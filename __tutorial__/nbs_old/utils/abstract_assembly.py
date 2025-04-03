import os
import pandas as pd
import random
import math


def subset_withholding(df: pd.DataFrame, t, features=[], withhold_interactions=[]):
    if not os.path.exists('./subsets'):
        os.makedirs('./subsets')

    remove = []

    for i in range(len(df)):
        row = df.iloc[i]

        interactions = set(withhold_interactions)

        if interactions.issubset(row.values.tolist()):
            remove.append(i)

    # print(remove)
    df = df.drop(index=remove, axis=0).reset_index().drop(
        columns=['index'], axis=1)
    df.to_csv(
        'subsets/abstract_subset-{}-t{}.csv'.format('_'.join(withhold_interactions), t))

    return


def subset_withholding_loop(df):
    df = df.copy()
    for a in A_levels:
        subset_withholding(df, 1, None, [a])
        for b in B_levels:
            subset_withholding(df, 1, None, [b])
            subset_withholding(df, 2, None, [a, b])
            for c in C_levels:
                subset_withholding(df, 1, None, [c])
                subset_withholding(df, 2, None, [b, c])
                subset_withholding(df, 2, None, [a, c])
                subset_withholding(df, 3, None, [a, b, c])
                for d in D_levels:
                    subset_withholding(df, 1, None, [d])
                    subset_withholding(df, 2, None, [a, d])
                    subset_withholding(df, 2, None, [b, d])
                    subset_withholding(df, 2, None, [c, d])
                    subset_withholding(df, 3, None, [a, b, d])
                    subset_withholding(df, 3, None, [a, c, d])
                    subset_withholding(df, 3, None, [b, c, d])
                    subset_withholding(df, 4, None, [a, b, c, d])

# SCRIPT SHOULD BE ONE TIME USE


A_levels = ['a{}'.format(i+1) for i in range(2)]
B_levels = ['b{}'.format(i+1) for i in range(2)]
C_levels = ['c{}'.format(i+1) for i in range(3)]
D_levels = ['d{}'.format(i+1) for i in range(4)]
E_levels = ['e{}'.format(i+1) for i in range(4)]

A_levels_skew = ['a1']*50 + ['a2']*5
B_levels_skew = ['b1']*25 + ['b2']*10
C_levels_skew = ['c1']*30 + ['c2']*20 + ['c3']*10
D_levels_skew = ['d1']*20 + ['d2']*10 + ['d3']*5 + ['d4']*5

labels = ['l1', 'l2', 'l3']

disc = False
df = pd.DataFrame(columns=['id', 'A', 'B', 'C', 'D', 'lab'])

i = 0
for a in A_levels:
    for b in B_levels:
        for c in C_levels:
            for d in D_levels:
                # for e in E_levels:
                for label in labels:
                    row = pd.Series({col: None for col in df.columns})
                    row['id'] = hash(i)
                    row['A'] = a
                    row['B'] = b
                    row['C'] = c
                    row['D'] = d
                    # row['E'] = e
                    row['lab'] = label
                    i += 1
                    df = df.append(row, ignore_index=True)

print(df)
df.to_csv('../datasets_tabular/abstract_fullfactorial.csv')
# subset_withholding_loop(df)

disc = True
df_skew = pd.DataFrame(columns=['id', 'A', 'B', 'C', 'D', 'lab'])
n = 1000
for i in range(n):
    row = pd.Series({col: None for col in df.columns})
    row['id'] = hash(i)
    row['A'] = random.sample(A_levels_skew, 1)[0]
    if disc:
        row['B'] = random.sample(B_levels_skew, 1)[0]
    else:
        row['B'] = round(random.uniform(0.0, 30.0), 3)
    row['C'] = random.sample(C_levels_skew, 1)[0]
    row['D'] = random.sample(D_levels_skew, 1)[0]
    row['lab'] = random.sample(labels, 1)[0]
    # i += 1
    df_skew = df_skew.append(row, ignore_index=True)

df_skew.to_csv('../datasets_tabular/abstract_skew.csv')

disc = False
df_random = pd.DataFrame(columns=['id', 'A', 'B', 'C', 'D', 'lab'])
n = 1000
for i in range(n):
    row = pd.Series({col: None for col in df.columns})
    row['id'] = hash(i)
    row['A'] = random.sample(A_levels, 1)[0]
    if disc:
        row['B'] = random.sample(B_levels, 1)[0]
    else:
        row['B'] = round(random.uniform(0.0, 30.0), 3)
    row['C'] = random.sample(C_levels, 1)[0]
    row['D'] = random.sample(D_levels, 1)[0]
    row['lab'] = random.sample(labels, 1)[0]
    # i += 1
    df_random = df_random.append(row, ignore_index=True)

df_random.to_csv('../datasets_tabular/abstract_native.csv')
