import os

from config import checks, config, splitperf
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
        self.split, self.split_id = config.extract_split_simple(codex_input)
        self.performance, self.split_id_perf, self.metrics = config.extract_perf_simple(
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
        self,
        coverage_subset="train",
        display_n=10,
        order="ascending descending",
        probing=False,
    ):
        # SORT, THEN RESET INDEX AND DROP
        train_df, val_df, test_df = dataset.df_slice_by_split_reorder(
            sample_id_col=self.sample_id_col,
            dataset_df=self.dataset_df,
            split=self.split,
        )

        performance_df = dataset.df_transpose_reorder_by_index(
            self.performance["test"]["Per-Sample Performance"]
        )

        coverage_results = results.stock_results_empty(
            self.dataset_name,
            self.model_name,
            self.strengths,
            self.universe,
            split_id=self.split_id,
            mode="performance by interaction",
        )

        coverage_results["results"] = combinatorial.performanceByInteraction_main(
            test_df,
            train_df,
            performance_df,
            self.dataset_name,
            self.universe,
            self.strengths,
            self.output_dir,
            self.metrics,
            sample_id=self.sample_id_col,
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
        coverage_results["info"]["Overall Performance"] = self.performance["test"].get(
            "Overall Performance",
            {"Overall Performance Statistics not found/incorrectly labeled."},
        )

        coverage_results = output.performance_by_interaction_vis(
            self.output_dir,
            self.dataset_name,
            coverage_results,
            self.strengths,
            self.metrics,
            order,
            display_n,
            subset=coverage_subset,
        )

        return coverage_results

    def balanced_test_set_construction(
        self,
        include_baseline=True,
        shuffle=False,
        adjusted_size=None,
        form_exclusions=False,
    ):
        """
        Runs test set post optimization in order to construct as balanced a test set as possible
        balancing appearance of various interactions. (???)

        Parameters
        - input: dict
            Read from the input file containing experiment requirements, pathing, info.
        - test_set_size_goal: int
            The desired number of samples for each balanced test set.

        Returns:
        - result: dict
            Coverage results on the overall dataset as well as split designations for each withheld
            interaction resulting from post-test set optimzation
        """

        coverage_results = results.stock_results_empty(
            self.dataset_name,
            self.model_name,
            self.strengths,
            self.universe,
            split_id=self.split_id,
            mode="balanced test construction",
        )

        if shuffle:
            dataset_df_used = self.dataset_df.sample(len(self.dataset_df))
        else:
            dataset_df_used = self.dataset_df

        if adjusted_size is None:
            coverage_results["results"] = combinatorial.balanced_test_set(
                dataset_df_used,
                self.dataset_name,
                self.sample_id_col,
                self.universe,
                self.strengths,
                self.test_set_size_goal,
                self.output_dir,
                include_baseline=include_baseline,
                form_exclusions=form_exclusions,
            )
        else:
            coverage_results = combinatorial.balanced_test_set(
                dataset_df_used,
                self.dataset_name,
                self.sample_id_col,
                self.universe,
                self.strengths,
                adjusted_size,
                self.output_dir,
                include_baseline=include_baseline,
                form_exclusions=form_exclusions,
            )

        if not os.path.exists(os.path.join(self.output_dir, "splits_by_json")):
            os.makedirs(os.path.join(self.output_dir, "splits_by_json"))
        if not os.path.exists(os.path.join(self.output_dir, "splits_by_csv")):
            os.makedirs(os.path.join(self.output_dir, "splits_by_csv"))

        for k in coverage_results["results"].keys():
            if "model_" in k:
                output.output_json_readable(
                    coverage_results["results"][k],
                    write_json=True,
                    file_path=os.path.join(
                        self.output_dir, "splits_by_json", k + ".json"
                    ),
                )

        output.output_json_readable(
            coverage_results,
            write_json=True,
            file_path=os.path.join(self.output_dir, "coverage.json"),
        )

        return coverage_results


test = CODEX(config_file="__tutorial_codex__/configs/config_rareplanes_example.json")
