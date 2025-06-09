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

    def parse_input_file(self, config_file:str):
        codex_input = config.handle_input_file(config_file)
        
        # Universal
        



        self.dataset_name, self.model_name = config.extract_names(codex_input)
        self.output_dir, self.strengths = config.define_experiment_variables(codex_input)
        self.universe, self.dataset_df = universing.define_input_space(codex_input)

        self.split, self.performance, self.metrics = config.extract_sp(codex_input)

        self.test_set_size_goal = codex_input.get('test_set_size_goal')
        
        return

    def dataset_evaluation(
        self):
        """
        Dataset evaluation comptues combinatorial coverage on a dataset.

        Parameters:
        codex_input: dict
            JSON config dictionary.
        """
        codex_input = self.cod

        checks.input_checks(codex_input)

        coverage_results = results.stock_results_empty(codex_input, self.universe)
        coverage_results["results"] = combinatorial.CC_main(
            self.dataset_df, self.dataset_name, self.universe, self.strengths, self.output_dir
        )

        coverage_results_formatted = output.dataset_eval_vis(
            self.output_dir, coverage_results
        )
        return coverage_results_formatted

    def dataset_split_evaluation(
        self,
        codex_input: dict,
        split: dict,
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
        checks.input_checks(codex_input, check_split=True)

        # produces a dataframe for each of the train/val/test subsets given the ids provided in the split file
        dataset_name, model_name = config.extract_names(codex_input)
        output_dir, strengths = config.define_experiment_variables(codex_input)
        universe, dataset_df = universing.define_input_space(codex_input)

        split_id = split["split_id"]
        sample_id_col = codex_input["sample_id_column"]

        train_df, val_df, test_df = dataset.df_slice_by_split_reorder(
            sample_id_col=sample_id_col, dataset_df=dataset_df, split=split
        )

        coverage_results = results.stock_results_empty(
            codex_input, universe, split_id=split_id
        )
        coverage_results["results_all"] = combinatorial.SDCC_main(
            train_df,
            source_name,
            test_df,
            target_name,
            dataset_name,
            universe,
            strengths,
            output_dir,
            comparison_mode=comparison,
            split_id=split_id,
        )
        if val_df is not None:
            coverage_results["results_all"].update(
                combinatorial.SDCC_main(
                    train_df,
                    source_name,
                    val_df,
                    "val",
                    dataset_name,
                    universe,
                    strengths,
                    output_dir,
                    comparison_mode=comparison,
                    split_id=split_id,
                )
            )

        coverage_results_formatted = output.dataset_split_eval_vis(
            output_dir,
            coverage_results,
            split_id,
            comparison=False,
        )
        return coverage_results_formatted

    def performance_by_interaction(
        self,
        codex_input,
        split,
        performance,
        metrics,
        coverage_subset="train",
        display_n=10,
        order="ascending descending",
        probing=False,
    ):
        """
        Codex mode that studies performance of each interaction in the dataset by aggregating
        per-sample performance of samples containing a particular interaction.

        Parameters:
        input: JSON
            Read from the input file containing experiment requirements, pathing, info.
        split: dict
            Read from a JSON file whose keys are the split partition and values are lists of
            the sample ID's as presented in the dataset.
        performance: dict
            Read from a JSON file containing per-sample model performance on a test set.
        metric: str
            Chosen metric to evaluate performance for the experiment.
        display_n: int
            Number of interactions to compare in each of the test set.
        order: str
            Display the n lowest interactions, or n highest interactions as appearing in the test set.

        Returns:
        results: dict
            Coverage results and human readable performance of the model on the test sets.

        Raises:
        KeyError
            If IDs of split file do not match those of per-sample performance section.
        """
        checks.input_checks(codex_input, check_split=True, check_perf=True)

        dataset_name, model_name = config.extract_names(codex_input)
        output_dir, strengths = config.define_experiment_variables(codex_input)
        universe, dataset_df = universing.define_input_space(codex_input)
        sample_id_col = codex_input["sample_id_column"]

        # SORT, THEN RESET INDEX AND DROP
        train_df, val_df, test_df = dataset.df_slice_by_split_reorder(
            sample_id_col=sample_id_col, dataset_df=dataset_df, split=split
        )

        performance_df = dataset.df_transpose_reorder_by_index(
            performance["test"]["Per-Sample Performance"]
        )

        coverage_results = results.stock_results_empty(codex_input)
        coverage_results["results"] = combinatorial.performanceByInteraction_main(
            test_df,
            train_df,
            performance_df,
            dataset_name,
            universe,
            strengths,
            output_dir,
            metrics,
            sample_id=sample_id_col,
            coverage_subset=coverage_subset,
        )
        # coverage_results_sdcc[t] = combinatorial.SDCC_main(trainDF, 'train', testDF, 'test', universe, t, output_dir, split_id=split_id)
        try:
            coverage_results["info"]["Overall Performance"] = performance["test"][
                "Overall Performance"
            ]
        except:
            print("No overall performance section found...")

        coverage_results = output.performance_by_interaction_vis(
            output_dir,
            dataset_name,
            coverage_results,
            strengths,
            metrics,
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
        output_dir, strengths = config.define_experiment_variables(self.input)
        universe, dataset_df = universing.define_input_space(self.input)

        if shuffle:
            dataset_df = dataset_df.sample(len(dataset_df))

        if adjusted_size is None:
            result = combinatorial.balanced_test_set(
                dataset_df,
                input["dataset_name"],
                input["sample_id_column"],
                universe,
                strengths,
                int(test_set_size_goal),
                output_dir,
                include_baseline=include_baseline,
                form_exclusions=form_exclusions,
            )
        else:
            result = combinatorial.balanced_test_set(
                dataset_df,
                input["dataset_name"],
                input["sample_id_column"],
                universe,
                strengths,
                int(adjusted_size),
                output_dir,
                include_baseline=include_baseline,
                form_exclusions=form_exclusions,
            )

        if not os.path.exists(os.path.join(output_dir, "splits_by_json")):
            os.makedirs(os.path.join(output_dir, "splits_by_json"))
        if not os.path.exists(os.path.join(output_dir, "splits_by_csv")):
            os.makedirs(os.path.join(output_dir, "splits_by_csv"))

        for k in result.keys():
            if "model_" in k:
                output.output_json_readable(
                    result[k],
                    write_json=True,
                    file_path=os.path.join(output_dir, "splits_by_json", k + ".json"),
                )
        result["universe"] = universe

        output.output_json_readable(
            result, write_json=True, file_path=os.path.join(output_dir, "coverage.json")
        )

        return result


test = CODEX(config_file='../__tutorial_codex__/configs/config_rareplanes_example.json')