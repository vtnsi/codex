# Format of Split and Performance Files
Currently, CODEX supports specific formatting of split files and performance files provided by the user.

## Split files
Split files specify the IDs of samples belonging to each portion of a dataset split in a dataset split. At minimum, CODEX requires the split to have `"train"` and `"test"` portions. A `"split_id"` key is used to keep track of the particular split used in the experiment.

```JSON
{
    "split_id": "split_x_",
    "train": ["img1", "img2", "img3", "..."],
    "validation": ["img74", "img75", "img76", "..."],
    "test": ["img88", "img99", "img100", "..."]
}
```

## Performance files
Performance files are dictionaries containing performance metrics from model evaluation computed prior to the experiment. 

Performance files contain information on split file from which model performance resulted and overall model performance on the test partition. For some CODEX modes, results of model inference at the sample level is required. This per-sample performance is stored under "Per Sample Performance" on the same level as "Overall Performance." `dataset split comparison` only requires overall performance, while `performance by interaction requires` per-sample performance.

In per-sample performance, sample ID's in this file must match those that identify samples in the tabular dataset. Any number of performance metrics can be stored under the "Overall Performance" and "Per-Sample Performance" fields.

Example: `performance_x_.json`
```JSON
{
    "split_file": "split_x_.json",
    "subset": "test": {
        "Overall Performance": {
            "precison": 0.5,
            "recall": 0.4,
            "f1": 0.444
        },
        "Per Sample Performance": {
            "img99": {
                "precision": 0.3, 
                "recall": 0.25
            },
            "img88": {
                "precision": 0.25,
                "recall": 0.64
            },
            "..."
        }
    }
}
```