import pandas as pd

df = pd.read_csv('dataset_rareplanes_lightweight_EX.csv', index_col=0)
df.sample(100).to_csv('dataset_rareplanes_lightweight-ex.csv', index=False)