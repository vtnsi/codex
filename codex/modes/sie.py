import torch
import os
import copy


class SIE:
    def __init__(self, sie_splits: dict, base_model: str, data_dir="", training_dir=""):
        self.sie_results = {split_id for split_id in sie_splits}

        self.sie_splits = sie_splits
        self.base_model = self.__load_model__(base_model)
        # self.__load_model__(base_model)

        self.data_dir = data_dir
        self.training_dir = training_dir

        print(self.base_model.keys(), type(self.base_model))
        print("Class safely created.")

        return

    def __load_model__(self, model_path: str):
        torch.serialization.add_safe_globals(["DetectionModel"])

        model = torch.load(model_path)

        """model = torch.nn.Module()
        model.load_state_dict(torch.load(f=os.path.realpath(model_path), weights_only=True))"""

        return model

    def __get_sie_model_path__(self, split_id):
        model_path = os.path.join(self.training_dir, f"{split_id}_t.pt")
        return model_path

    def __save_trained_model__(self, model, split_id):
        model_path = self.__get_sie_model_path__(self, split_id)
        torch.save(model.state_dict(), model_path)
        self.sie_results[split_id]["trained_model_path"] = model_path

    def sie_train(self, train_func, select_split=None):
        for split_id in self.sie_splits:
            model = copy.deepcopy(self.base_model).train()
            model = model.train_func()

            data = None

            self.__save_trained_model__(model, split_id=split_id)

        pass

    def sie_eval(self, eval_func, metrics, select_split=None):
        for split_id in self.sie_splits():
            model_path = self.__get_sie_model_path__(self, split_id)
            model = self.__load_model__(model_path).eval()

            data = None

            results = model.eval_func()
            self.sie_results[split_id]["results"] = results

    def sie_analyze(self):
        return


def systematic_inclusion_exclusion(
    sie_splits: dict, base_model: str, data_dir="", training_dir=""
):
    return
