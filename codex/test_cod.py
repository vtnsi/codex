from codex import CODEX

test = CODEX(
    config_file="__tutorial_codex__/configs/config_rareplanes_example_bt.json"
)
test.balanced_test_set_construction()
exit()

test.performance_by_interaction()
test.dataset_split_evaluation()
