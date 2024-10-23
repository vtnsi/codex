# Modes
CODEX offers several modes of dataset exploration applicable to any tabular dataset.
### Dataset Evaluation
**Dataset evaluation** computes combinatorial coverage over a dataset with respect to a defined universe.
- Mode keyword: `dataset evaluation`

### Dataset Split Evaluation
**Dataset split evaluation** computes set difference combinatorial coverage between two portions of a single dataset split with respect to a defined universe.
- Mode keyword: `dataset split evaluation`

### Dataset Split Comparison
**Dataset split comparison** relates set difference combinatorial coverage to model performance on dataset splits between multiple splits and their corresponding performance files.
- Mode keyword: `dataset split comparison`

### Performance by Interaction
**Performance by interaction** computes the aggregate performance for each interaction appearing in a dataset by leveraging the per-sample performance of test samples that contain the interaction.
- Mode keyword: `performance by interaction`

### Balanced set construction
**Balanced set construction** attempts to construct a subset of a dataset given a goal number of samples such that the interactions appear in the dataset as balanced (equally) as possible. Perfect balance may not always be possible given constraints in the original dataset. It builds the set in two phases. The first phase builds the subset one sample at a time by prioritizing placing rarely appearing interactions first. The second phase conducts post-optimization by checking to see if samples that contain highly redundant interactions can be removed without dropping any interaction below the goal.
- Mode keyword: `balanced set construction`

## Modes in Development
Additional modes for CODEX are in the process of being added.
### Systematic Inclusion/Exclusion (SIE)
**Systematic Inclusion/Exclusion** aims to uncover the impact on model training that a $t$-way interaction has for each interaction appearing in a dataset for each specified $t$. This mode does so by constructing a universal test set and from it, systematically generates a collection of dataset splits that each specify a test set that withholds the particular $t$-way interaction (test included) and another test set that only contains samples with that interaction (test excluded). Then, a chosen model automatically and independently trained and evaluated on each split to test for differential performance when the interaction is present or absent in the test set.
- Mode keyword: `systematic inclusion exclusion`

### Data Set Differencing
**Dataset set differencing** computes SDCC on any two portions of a dataset or even two datasets with respect to a defined and shared universe.
- Mode keyword: `dataset data set dfiferencing`

### Dataset Set Difference Sample Selection
**Dataset set differencing sample selection** identifies the ID's of samples that exist in the set difference when computing the set difference of interactions by computing SDCC between two portions of a dataset or two datasets with respect to a defined and shared universe.
- Mode keyword: `dataset data set difference sampling`

### Model Probing
**Model probing** performs performance by interaction on two independent test sets - a probing test set and an expoitation test set - to evaluate areas of model robustness or inconsistency by identifying similarities and differences between the low-performing and high-performing interactions in the probing and exploitation sets, respectively.
- Mode keyword: `model probing`
