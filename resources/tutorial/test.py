import pandas as pd
pd.read_csv('dataset_abstract_EX.csv').drop(['Unnamed: 0'], axis=1).to_csv('dataset_abstract_EX.csv', index=False)
exit()
df = pd.read_csv('dataset_rareplanes_lightweight_EX.csv', index_col=0)
df.sample(100).to_csv('dataset_rareplanes_lightweight-ex.csv', index=False)