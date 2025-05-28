import pandas as pd
import json

from sklearn.model_selection import train_test_split
from codex.modules import output

df = pd.read_csv("dataset_abstract.csv", index_col="id")

random = False
if random:
    train, test_val = train_test_split(df, test_size=0.4)
    val, test = train_test_split(test_val, test_size=0.5)
else:
    print(df)
    df = df.sort_values(by="A", axis=0)
    print(df)

    train = df[: int(0.7 * len(df))]
    val = df[int(0.7 * len(df)) : int(0.15 * len(df))]
    test = df[int(0.15 * len(df)) :]

save = {
    "split_id": "abstract_example_01-sortedA",
}
train_ids = train.index.tolist()
val_ids = val.index.tolist()
test_ids = test.index.tolist()
train_ids = list(map(str, train_ids))
val_ids = list(map(str, val_ids))
test_ids = list(map(str, test_ids))
print(test_ids)
save["train"] = train_ids
save["val"] = val_ids
save["test"] = test_ids

output.output_json_readable(
    save, write_json=True, file_path="split_abstract-sortedA.json"
)


exit()
pd.read_csv("dataset_abstract_EX.csv").drop(["Unnamed: 0"], axis=1).to_csv(
    "dataset_abstract_EX.csv", index=False
)
exit()
df = pd.read_csv("dataset_rareplanes_lightweight_EX.csv", index_col=0)
df.sample(100).to_csv("dataset_rareplanes_lightweight-ex.csv", index=False)
