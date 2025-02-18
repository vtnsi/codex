import os
import sys
import json
import yaml
import shutil
import glob
import copy
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy
import torch
import torchvision
import torchmetrics
import cv2
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.plotting import plot_results
from ultralytics.utils.plotting import plt_color_scatter
from collections import OrderedDict

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torchvision
from multiprocessing import Process, Pool

from ultralytics import YOLO
from ultralytics.utils.plotting import plt_color_scatter
from ultralytics.utils.plotting import plot_results

import typing_extensions
import importlib

verbose = False


def json_pretty_write(json_obj, file_path, sort=False):
    json_str = json.dumps(json_obj, sort_keys=True, indent=4, separators=(",", ": "))
    with open(file_path, "w") as f:
        f.write(json_str)

    return json_str



