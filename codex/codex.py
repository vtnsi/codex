import os

from inputs import checks, config, splitperf
from universe import universing, dataset
from combinatorial import combinatorial
from output import output, results
from modes import sie_yolo_ml as sie_ml, sie_analysis

# Brian Lee
# Refactored for class format ()
from modes.sie import SIE


class CODEX:
    def __init__(self, config_file=None, verbose=1):
        """
        CODEX class constructor. The CODEX class serves as the hub of all possible
        experiments for a given dataset of interest and the necessary variables of
        interest depending on the experiment. The CODEX class links the input and
        output with combinatorial methods of exploration in between.

        Parameters:
            config_file (str|dict): Config dictionary or path to config file containing specifications
                for CODEX experiments. While there are inputs required by all modes, certain modes
                are only possible insofar as the necessary inputs are included in this file.

            verbose (int): Level of verbosity for experiment output (default=1), written to the logs and/or console.
                verbose=0: Essential and informative outputs only.
                verbose=1: Metrics, variables, important paths displayed.
                verbose=2: Primarily for debugging - connective variables, data
                    structure info , etc.
        """
        self.config_file = config_file
        self.verbose = str(verbose)

        self._parse_input_file(config_file)

    def _parse_input_file(self, config_file: str):
        """
        Stores config file variables into CODEX object in memory.

        Args:
            config_file (str|dict): Config dictionary or path to config file containing specifications
                for CODEX experiments.
        """

        codex_input = config.handle_input_file(os.path.realpath(config_file))
        checks.input_checks(codex_input)

        # Variables direct access
        self.config_id = codex_input["config_id"]
        self.strengths = codex_input["t"]
        self.sample_id_col = codex_input["sample_id_column"]

        # Variables direct access, but mode-dependent
        self.test_set_size_goal = codex_input.get("test_set_size_goal")
        self.training_params = codex_input.get("training_params", {})

        # Variables requiring extraction/preprocessing
        self.output_dir = output.create_output_dir(
            os.path.realpath(
                os.path.join(
                    codex_input.get("output_dir"), codex_input.get("config_id")
                )
            ),
            timed=codex_input.get("timestamp", False),
        )
        self.dataset_name, self.model_name = config.extract_names(codex_input)
        self.universe, self.dataset_df = universing.define_input_space(codex_input)

        # Variables requiring extraction, mode-dependent
        self.split, self.split_id = splitperf.extract_split_simple(codex_input)
        self.performance, self.split_id_perf, self.metrics = (
            splitperf.extract_perf_simple(codex_input)
        )

        self.split_dir = ""

        return True

    def dataset_evaluation(self):
        """
        Computes combinatorial coverage on a dataset with respect to a defined universe.
        """

        coverage_results = results.stock_results_empty(
            self.dataset_name,
            self.model_name,
            self.strengths,
            self.universe,
            mode="dataset evalutation",
        )
        coverage_results["results"] = combinatorial.CC_main(
            self.dataset_df,
            self.dataset_name,
            self.universe,
            self.strengths,
            self.output_dir,
        )

        coverage_results_formatted = output.dataset_eval_vis(
            self.output_dir, coverage_results
        )
        return coverage_results_formatted

    def dataset_split_evaluation(
        self,
        source_name="train",
        target_name="test",
        comparison=False,
    ):
        """
        Computes set difference combinatorial coverage between two portions of a
        dataset with respect to a defined universe.

        Args:
            source_name (str): Name of portion designated as the "source" dataset in the split file specified.
            target_name (str): Name of portion designated as the "target" dataset in the split file specified.
            comparison (bool) (default=False): Flags for dataset split comparison.
        """
        train_df, val_df, test_df = dataset.df_slice_by_split_reorder(
            sample_id_col=self.sample_id_col,
            dataset_df=self.dataset_df,
            split=self.split,
        )

        coverage_results = results.stock_results_empty(
            self.dataset_name,
            self.model_name,
            self.strengths,
            self.universe,
            split_id=self.split_id,
            mode="dataset split evaluation",
        )

        coverage_results["results_all"] = combinatorial.SDCC_main(
            train_df,
            source_name,
            test_df,
            target_name,
            self.dataset_name,
            self.universe,
            self.strengths,
            self.output_dir,
            comparison_mode=comparison,
            split_id=self.split_id,
        )
        if val_df is not None:
            coverage_results["results_all"].update(
                combinatorial.SDCC_main(
                    train_df,
                    source_name,
                    val_df,
                    "val",
                    self.dataset_name,
                    self.universe,
                    self.strengths,
                    self.output_dir,
                    comparison_mode=comparison,
                    split_id=self.split_id,
                )
            )

        coverage_results_formatted = output.dataset_split_eval_vis(
            self.output_dir,
            coverage_results,
            self.split_id,
            comparison=False,
        )
        return coverage_results_formatted

    def performance_by_interaction(self, coverage_subset="train"):
        """
        Computes per-interaction performance using the samples containing said interaction
        each of t-way interactions as defined in the universe.

        Args:
            coverage_subset (str, default="train"): Key designating what portion of the split 
                to compute coverage and subsequent performance by interaction over.

                ** amend above ^^ 06.24.25
        """

        # SORT, THEN RESET INDEX AND DROP
        train_df, val_df, test_df = dataset.df_slice_by_split_reorder(
            sample_id_col=self.sample_id_col,
            dataset_df=self.dataset_df,
            split=self.split,
        )

        ps_performance_df = dataset.df_transpose_reorder_by_index(
            self.performance["test"]["Per-Sample Performance"]
        )

        coverage_results = results.stock_results_empty(
            self.dataset_name,
            self.model_name,
            self.strengths,
            self.universe,
            split_id=self.split_id,
            mode="performance by interaction",
            metrics=self.metrics,
            coverage_subset=coverage_subset,
        )
        coverage_results["info"]["Overall Performance"] = self.performance["test"].get(
            "Overall Performance",
            {"Overall Performance Statistics not found/incorrectly labeled."},
        )

        coverage_results["results"] = combinatorial.performanceByInteraction_main(
            train_dataDF=train_df,
            test_dataDF=test_df,
            performance_df=ps_performance_df,
            dataset_name=self.dataset_name,
            universe=self.universe,
            strengths=self.strengths,
            output_dir=self.output_dir,
            metrics=self.metrics,
            coverage_subset=coverage_subset,
        )

        sdcc = False
        if sdcc:
            coverage_results["results_sdcc"] = combinatorial.SDCC_main(
                train_df,
                "train",
                test_df,
                "test",
                self.dataset_name,
                self.universe,
                self.strengths,
                self.output_dir,
                False,
                self.split_id,
            )

        coverage_results = output.performance_by_interaction_vis(
            self.output_dir,
            coverage_results,
            coverage_subset=coverage_subset,
            display_interaction_num=10,
            display_interaction_order="ascending descending",
        )

        print("reached")
        return coverage_results

    def balanced_test_set_construction(
        self, include_baseline=True, construct_splits=False
    ):
        """
        Runs a <> algorithm to balance a universal test set in terms of frequency of 
        t-way interactions appearing for a defined universe as best as possible.
        
        Args:
            construct_splits (bool, default=False): Whether or not to generate and write split 
                files withholding a particular t-way interaction from test for each interaction for
                the largest t possible.
            include_baseline (bool, default=True): Whether or not to generate a random split 
                that serves as a baseline split in addition to those withholding interactions.

                Does nothing if construct_splits is False.
        """

        coverage_results = results.stock_results_empty(
            self.dataset_name,
            self.model_name,
            self.strengths,
            self.universe,
            split_id=self.split_id,
            mode="balanced test construction",
            test_set_size_goal=self.test_set_size_goal,
        )

        dataset_df = self.dataset_df.sample(len(self.dataset_df))

        coverage_results["results"] = combinatorial.balanced_test_set(
            dataset_df,
            self.dataset_name,
            self.sample_id_col,
            self.universe,
            self.strengths,
            self.test_set_size_goal,
            self.output_dir,
            include_baseline=include_baseline,
            construct_splits=construct_splits,
        )

        output.output_json_readable(
            coverage_results,
            write_json=True,
            file_path=os.path.join(self.output_dir, "coverage.json"),
        )

        return coverage_results

    # ~~~~~~~~~~~~~~~~~~~ Dependent modes ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def dataset_split_comparison(
        self,
        codex_input: dict,
        split_multiple: dict,
        performance_multiple: dict,
        metric: str,
    ):
        """
        Dataset split comparison compares SDCC values from multiple splits
        with their resultant model performances agaisnt eachother by calling
        dataset split evaluation for each split.

        Split files required to contain at least train and test splits.

        Parameters
        input: JSON
            Read from the input file containing experiment requirements, pathing, info.

        split_list: list
            A list of file paths pointing to JSON files whose keys are the split partition
            and values are lists of the sample ID's as presented in the dataset.

        performance_list: list
            A list of JSON files containing model performance expressed as a
            named metric. References the split file used to obtain model
            performance. Only overall model performance is required for this mode.

        metric: str
            Chosen metric to evaluate performance for the experiment.

        sdcc_direction: str
            Direction of the set difference between the target and source datasets.
        """
        split_ids = [split_multiple[split]["split_id"] for split in split_multiple]
        coverage_results = results.stock_results_empty(
            codex_input,
            self.dataset_name,
            self.model_name,
            self.universe,
            split_id=split_ids,
        )
        for split_file in split_multiple:
            split_id = split_multiple[split_file]["split_id"]
            coverage_results[split_id] = self.dataset_split_evaluation(
                input, split_multiple[split_file], comparison=True
            )
        if performance_multiple is None:
            return coverage_results

        coverage_results = output.dataset_split_comp_vis(
            self.output_dir,
            self.dataset_name,
            coverage_results,
            performance_multiple,
            self.strengths,
            metric,
            split_ids=split_ids,
        )
        return coverage_results

    def model_probe(
        codex_input,
        split_p,
        split_e,
        perf_p,
        perf_e,
        metric,
        name_p="_included",
        name_e="_excluded",
        display_n=10,
        order="ascending descending",
    ):
        """
        Model probing computes performance by interaction on two partitions of a test set - a probing set
        and exploit set - to compare how high-performing and low-performing interactions of the probing set
        perform in a potential exploit set and highlight the model's vulnerabilities.

        Parameters
        - input: JSON
            Read from the input file containing experiment requirements, pathing, info.
        - split_p: dict
            Read from a JSON file whose keys are the split partition and values are lists of
            the sample ID's as presented in the dataset, with the probing partition.
        - performance_p: dict
            Read from a JSON file containing overall model performance on the probing partition
            of the test set as well as per-sample performance performed by the model on each sample
            in the probe set.
        - split_e: dict
            Read from a JSON file whose keys are the split partition and values are lists of
            the sample ID's as presented in the dataset, with the exploit partition.
        - performance_e: dict
            Read from a JSON file containing overall model performance on the exploit partition
            of the test set as well as per-sample performance performed by the model on each sample
            in the exploit set.
        - metric: str
            Chosen metric to evaluate performance for the experiment.
        display_n: int
            Number of interactions to compare in each of the probe and exploit sets.
        order: str
            Display the n lowest interactions, or n highest interactions as appearing in the probe set.

        Returns:
        - results: dict
            Coverage results and human readable performance of probing and exploit sets.
        """
        checks.checks_generic(codex_input)
        checks.checks_split(codex_input)
        checks.checks_perf(codex_input)

        output_dir, strengths = config.define_experiment_variables(input)
        probe_coverage = performance_by_interaction(
            input, split_p, perf_p, metric, subset=name_p, withhold=True
        )
        exploit_coverage = performance_by_interaction(
            input, split_e, perf_e, metric, subset=name_e, withhold=True
        )

        result = output.model_probing_vis(
            output_dir,
            name_p,
            name_e,
            probe_coverage,
            exploit_coverage,
            strengths,
            metric,
            order,
            display_n,
            subset="train",
        )
        return result

    def systematic_inclusion_exclusion(
        self,
        model,
        training_params,
        test_set_size_goal,  # codex_input, test_set_size_goal, training_params=None
        balance_test_sets=True,
        overwrite=False,
    ):
        # Check split dir: If not containing SIE splits, run balanced test regardless, extract split dir and set to self.split_dir.
        # If want to, rerun balanced test sets, new self.split_dir = to

        results = self.balanced_test_set_construction(
            include_baseline=True, construct_splits=True
        )
        sets = results["results"]["split"]
        self.split_dir = results["results"]["split_dir_json"]

        # If already ran: point to correct SIE split directory or JSON
        sie_splits = None; data_dir = None; training_dir = None; base_model = None
        sie_runner = SIE(sie_splits, base_model, data_dir, training_dir)
        
        sie_runner.sie_train()
        sie_runner.sie_eval()
        sie_runner.sie_analyze()

        return sets

    def performance_by_frequency_coverage(
        self, codex_input, skew_levels: list, test_set_size_goal=250
    ):
        results_sets = self.balanced_test_set_construction(
            include_baseline=True, construct_splits=True
        )
        sets = results["results"]["split"]
        self.split_dir = results["results"]["split_dir_json"]

        results_biasing = combinatorial.performance_by_frequency_coverage_main()

        output.output_json_readable(
            results_biasing,
            write_json=True,
            file_path=os.path.join(self.output_dir, "pbcf.json"),
        )

        return results_biasing
