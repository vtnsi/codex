import sie
from ultralytics import YOLO
import torch
from torch.nn import Conv1d
import json
import sie

model = Conv1d(in_channels=3, out_channels=1, kernel_size=3)
torch.save(model.state_dict(), "conv.pt")

# model = YOLO('yolo11n.pt')

with open(
    "/home/hume-users/leebri2n/PROJECTS/dote_1070-1083/CODEX_NSI_MASTER/__tutorial_codex__/runs/EXAMPLE-bt_rareplanes_ref/coverage.json"
) as f:
    sie_splits = json.load(f)["results"]["splits"]

sie_test = sie.SIE(sie_splits, "conv.pt", "", "")
sie_test.sie_train(None)

exit()
sie.systematic_inclusion_exclusion(sie_splits, "yolo11n.pt", "", "")
