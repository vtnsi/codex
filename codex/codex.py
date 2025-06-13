import os

from inputs import checks, config, splitperf
from universe import universing, dataset
from combinatorial import combinatorial
from output import output, results
import modes

# Brian Lee
# Refactored for class format ()


class CODEX:
    def __init__(self, config_file=None, verbose="1"):
        self.config_file = config_file
        self.verbose = verbose

        self.parse_input_file(config_file)

    def parse_input_file(self, config_file: str):
        codex_input = config.handle_input_file(config_file)
        checks.input_checks(codex_input)

        # Variables direct access
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
        self.performance, self.split_id_perf, self.metrics = splitperf.extract_perf_simple(
            codex_input
        )

        return True

    def dataset_evaluation(self):
        """
        Dataset evaluation comptues combinatorial coverage on a dataset.

        Parameters:
        codex_input: dict
            JSON config dictionary.
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
        Dataset split evaluation computes SDCC from a split and plots it against its resultant
        model performance. Split file required to contain at least train and test splits.

        Parameters
        input: dict
            Read from the input file containing experiment requirements, pathing, info.

        split: dict
            Read from a JSON file whose keys are the split partition and values are lists of
            the sample ID's as presented in the dataset.

        performance: dict
            Read from a JSON file containing a model's performance on a test set.

        metric: str
            Chosen metric to evaluate performance for the experiment.

        sdcc_direction: str
            Direction of the set difference between the target and source datasets.
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

    def performance_by_interaction(
        self, coverage_subset="train", display_n=10, order="ascending descending"
    ):
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
            performanceDF=ps_performance_df,
            name=self.dataset_name,
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
            order,
            display_n,
            coverage_subset=coverage_subset,
        )

        print("reached")
        return coverage_results

    def balanced_test_set_construction(
        self,
        include_baseline=True,
        adjusted_size=None,
        form_exclusions=False,
    ):
        """ """

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
            form_exclusions=form_exclusions,
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
        codex_input, test_set_size_goal, training_params=None
    ):
        output_dir, strengths = config.define_experiment_variables(codex_input)

        data_dir = codex_input["original_data_directory"]
        dataset_dir_YOLO = codex_input["dataset_directory"]
        training_dir = codex_input["training_directory"]
        if dataset_dir_YOLO == "":
            if not os.path.exists(os.path.join(output_dir, "datasets")):
                os.makedirs(os.path.join(output_dir, "datasets"))
            dataset_dir_YOLO = os.path.join(output_dir, "datasets")

        result = None

        if os.path.exists(os.path.join(output_dir, "coverage.json")):
            overwrite_splits = (
                input(
                    "Balanced test set splits already found for {}. Overwrite splits? (y/n): ".format(
                        output_dir
                    )
                )
                == "y"
            )
            if overwrite_splits:
                shutil.rmtree(output_dir)
                result = balanced_test_set_construction(codex_input, test_set_size_goal)
        else:
            result = balanced_test_set_construction(codex_input)

        with open(os.path.join(output_dir, "coverage.json")) as f:
            result = json.load(f)

        n_t = result["universal_test_set_size"]
        n_g = codex_input["test set size goal"]
        proceed_wsize = (
            input(
                "Assembled a test set of {} samples compared to goal of {}. Proceed? (y/n): ".format(
                    n_t, n_g
                )
            )
            == "y"
        )
        if not proceed_wsize:
            adjusted_size = input("New test set goal (int): ")
            result = balanced_test_set_construction(
                codex_input, adjusted_size=adjusted_size
            )

        with open(os.path.join(output_dir, "coverage.json")) as f:
            result = json.load(f)

        if result is None:
            with open(os.path.join(output_dir, "coverage.json")) as f:
                result = json.load(f)
        SIE_splits = {
            key.split("model")[-1]: result[key]
            for key in result.keys()
            if "model" in key
        }
        SIE_ids = list(SIE_splits.keys())
        print(SIE_ids)

        # CONSIDER ASSIGNING DATASET ACCORDING TO EACH CONFIG DIRECTORY
        train = False
        if train:
            pass
            sie_ml.data_distribution_SIE(
                SIE_splits, data_dir, dataset_dir_YOLO, overwrite=True, mode_text=True
            )
            sie_ml.train_SIE(
                SIE_splits,
                dataset_dir_YOLO,
                training_dir,
                epochs=300,
                batch=128,
                devices=[0, 1, 2, 3],
                force_resume=True,
            )

        score = True
        if score:
            sie_ml.evaluate(
                SIE_splits,
                data_dir,
                dataset_dir_YOLO,
                training_dir,
                config_dir=output_dir,
            )
        table_filename = ""
        analyze = True

        if analyze:
            perf_table = pd.read_csv(
                os.path.join(
                    codex_input["codex_directory"],
                    codex_input["performance_folder"],
                    table_filename,
                )
            )
            metric = codex_input["metric"]
            metrics = codex_input["metrics"]
            features = codex_input["features"]

            output_dir, strengths = config.define_experiment_variables(codex_input)

            results = {}
            if metric == "all":
                for metric in metrics:
                    model_summary, contrasts_summary, contrast_names = (
                        sie_analysis.SIE_binomial_regression_main(
                            perf_table, metrics, metric, features
                        )
                    )
                    results = results.update(
                        output.SIE_regression_test_vis(
                            output_dir, model_summary, contrasts_summary, contrast_names
                        )
                    )
            else:
                model_summary, contrasts_summary, contrast_names = (
                    sie_analysis.SIE_binomial_regression_main(
                        perf_table, metrics, metric, features
                    )
                )
                results = output.SIE_regression_test_vis(
                    output_dir, model_summary, contrasts_summary, contrast_names
                )

            return SIE_splits, results

        return SIE_splits, None

    def systematic_inclusion_exclusion_binomial_linreg(codex_input, table_filename):
        return results

    def performance_by_frequency_coverage(
        codex_input, skew_levels: list, test_set_size_goal=250
    ):
        import modes.pbfc_biasing as biasing

        output_dir, strengths = config.define_experiment_variables(input)
        universe, dataset_df_init = universing.define_input_space(input)

        result = balanced_test_set_construction(
            codex_input, test_set_size_goal, form_exclusions=False
        )

        for t in strengths:
            results_all_models = combinatorial.performance_by_frequency_coverage_main()

        output.output_json_readable(
            results_all_models,
            write_json=True,
            file_path=os.path.join(output_dir, "pbcf.json"),
        )

        return results_all_models
