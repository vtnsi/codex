# Input File Parameters

## Required for all CODEX modes:
- `mode`: String of the requested mode of exploration to perform on a dataset.
  - `[dataset evaluation, dataset split evaluation, dataset split comparison, performance by interaction]`
- `codex_directory`: String specifying the directory containing all materials for a CODEX application.
  - A complete CODEX directory contains input files, datasets, binning files, split files, performance files, and experiment outputs.
- `config_dir`: String specifying the sub-directory in `codex_directory` containing experiment outputs including plots, metrics, logging files.
- `dataset_name`: String of name of dataset.
- `model_name`: String of name of machine learning model used for modes involving performance.
- `data_directory`: String of sub-directory in `codex_directory` containing tabular datasets.
- `dataset_file`: String of tabular dataset filename (`.csv`) within `data_directory`.
- `features`: List of strings of feature names used to construct the universe over as they appear in the tabular dataset.
- `sample_id_column`: String of column name containing IDs of samples as they appear in the tabular dataset.
- `label_column`: String of column name containing labels or classes of samples as they appear in the tabular dataset.
- `bin_directory`: String of sub-directory in `codex_directory` containing binning scheme files.
- `bin_file`: String of binning filename (`.txt.`) within `bin_directory`.
- `universe`: String of pathway to universe filename (JSON) within `codex_directory` if the user wishes to use a pre-computed universe. This overrides the "binning file" and "learning from dataset" options in universe construction.
- `use_augmented_universe`: Boolean allowing CODEX to expand a universe beyond the one constructed if encountering other features and values in the dataset.
- `counting_mode`: Boolean to include `label_column` as a feature in constructing the universe.
  - `"label_exclusive"` (default) / `"label_inclusive"`
- `t`: List of integers of $t$-way combinatorial strengths of interactions to compute combinatorial coverage over.
- `timed_output`: Boolean to tag logging files with date and time of execution. Default: `true`

## Required for specific CODEX modes: 
- `split_folder`: String of sub-directory in `codex_directory` containing split files.
- `split_file`: String of split filename (JSON) specifying the IDs of sample IDs as they appear in the dataset. See [formatting](split_performance_formatting.md).
  - OR: List of strings of multiple split filenames (JSON) to use in mode `"dataset split comparison"`
- `performance_folder`: String of sub-directory in `codex_directory` containing performance files.
- `performance_file`: String of performance filename (JSON) containing performance metrics resulting from a model on the dataset of interest. See [formatting](split_performance_formatting.md).
  - OR: List of strings of multiple corresponding performance filenames (JSON) to use in mode `"dataset split comparison"`
