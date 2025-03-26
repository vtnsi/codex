import pandas as pd
import json

from sklearn.model_selection import train_test_split
from codex.modules import output

df = pd.read_csv('dataset_rareplanes_lightweight.csv', index_col='image_tile_id')

train, test_val = train_test_split(df, test_size=0.4)
val, test = train_test_split(test_val, test_size=0.5)

save = {
    'split_id': 'rareplanes_example_01-random',
}
save['train'] = train.index.tolist()
save['val'] = val.index.tolist()
save['test'] = test.index.tolist()

output.output_json_readable(save, write_json=True, file_path='split_rareplanes_example.json')



exit()
pd.read_csv('dataset_abstract_EX.csv').drop(['Unnamed: 0'], axis=1).to_csv('dataset_abstract_EX.csv', index=False)
exit()
df = pd.read_csv('dataset_rareplanes_lightweight_EX.csv', index_col=0)
df.sample(100).to_csv('dataset_rareplanes_lightweight-ex.csv', index=False)