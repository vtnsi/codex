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
importlib.reload(typing_extensions)
'''print("PyTorch ver: ", torch.__version__)
print("TorchVision ver: ", torchvision.__version__)'''


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

'''try:
    from ..modules import output
except:
    from codex.modules import output'''

def __compute_bbox(annotation_df_selection, data_dir, tile_id):
    verbose=False
    box_id = 0
    label_path = os.path.join(data_dir, 'data', f'{tile_id}.txt')
    with open(label_path, 'w') as wf:
        with open(os.path.join(label_path), 'a') as wfo:
            for i, sample in annotation_df_selection.iterrows():
                role_id = str(int(sample['role_id'] - 1))
                x1 = sample['x1']
                y1 = sample['y1']
                width = sample['x2']
                height = sample['y2']
                xc = width/2 + x1
                yc = height/2 + y1
                coord = [xc, yc, width, height]
                coord_norm_str = [str(c / 512) for c in coord]
                
                label = [role_id, coord_norm_str[0], coord_norm_str[1], coord_norm_str[2], coord_norm_str[3]]#, str(box_id)]
                label_str = " ".join(label)
                label_str = f"{label_str}\n"
                wfo.write(label_str)
                box_id += 1
                if verbose:
                    print(f"{tile_id}:", label_str)

    return label_path

def construct_labels(data_dir):
    metadata_dir = os.path.realpath(os.path.join(data_dir, 'metadata'))
    metadata_df_orig = pd.read_csv(os.path.join(metadata_dir, 'RarePlanes_Public_Metadata.csv'))
    annotation_df = pd.read_csv(os.path.join(metadata_dir, 'bbox_annotations.csv'))

    tile_ids = annotation_df['image_tile_id'].unique()
    label_filenames = []

    for tile_id in tqdm(tile_ids):
        #img_id = tile_ids.split('_tile')[0]
        #subset = 'train' if metadata_df_orig[metadata_df_orig['image_id'] == img_id].iloc[0]['Train'] == 1 else 'test'
        annotation_df_selection = annotation_df[annotation_df['image_tile_id'] == tile_id]
        label_filename = __compute_bbox(annotation_df_selection=annotation_df_selection, data_dir=data_dir, tile_id=tile_id)
        label_filenames.append(label_filename)

    return label_filenames # realpaths of labels

def __write_yml_cls(wf):
    wf.write("names:\n")
    wf.write("  0: Small Civil Transport/Utility\n")
    wf.write("  1: Medium Civil Transport/Utility\n")
    wf.write("  2: Large Civil Transport/Utility\n")
    wf.write("  3: Military Transport/Utility/AWAC\n")
    wf.write("  4: Military Bomber\n")
    wf.write("  5: Military Fighter/Interceptor/Attack\n")
    wf.write("  6: Military Trainer\n")

def __yolo_img_specification_txt(data_img_dir, data_config_subdir, ids:list, exp_id, split_name:str, data_dir_outer=None):
    out_file = os.path.realpath(os.path.join(data_config_subdir, f'{exp_id}_{split_name}.txt'))
    with open(out_file, 'w') as f:
        for i, tile_id in enumerate(ids):
            label_path =os.path.realpath(os.path.join(data_img_dir, f'{tile_id}.txt'))

            if not os.path.exists(label_path):
                print("At least one label missing from desired dataset. Reconstructing labels.")
                construct_labels(data_dir_outer)
                
            f.write(str(os.path.realpath(os.path.join(data_img_dir, f'{tile_id}.png'))))
            if i != len(ids)-1:
                f.write('\n')

    return out_file

def __yolo_yaml_assembly(data_config_subdir, exp_id, train_samples_filename, val_samples_filename, test_samples_filename=None):
    out_file = os.path.realpath(os.path.join(data_config_subdir, f'dataset_detect-{exp_id}.yaml'))

    with open(out_file, 'w') as f:
        f.write(f"train: {train_samples_filename}\n")
        f.write(f"val: {val_samples_filename}\n")
        if test_samples_filename is not None:
            f.write(f"test: {test_samples_filename}\n")

        f.write('\n')
        __write_yml_cls(f)

    return out_file


def yolo_dataset(experiment_id, data_config_dir, data_dir, train_ids, val_ids, test_ids):
    data_image_dir = os.path.realpath(os.path.join(data_dir, 'data'))
    data_metadata_dir = os.path.realpath(os.path.join(data_dir, 'metadata'))

    data_config_subdir = os.path.realpath(os.path.join(data_config_dir, experiment_id))
    os.makedirs(data_config_subdir, exist_ok=True)

    train_samples_filename = __yolo_img_specification_txt(data_image_dir, data_config_subdir, train_ids, experiment_id, 'train', data_dir_outer=data_dir)
    val_samples_filename = __yolo_img_specification_txt(data_image_dir, data_config_subdir, val_ids, experiment_id, 'val', data_dir_outer=data_dir)
    test_samples_filename = __yolo_img_specification_txt(data_image_dir, data_config_subdir, test_ids, experiment_id, 'test', data_dir_outer=data_dir)
    
    dataset_config_filename = __yolo_yaml_assembly(data_config_subdir, experiment_id, train_samples_filename, val_samples_filename, test_samples_filename)

    return dataset_config_filename # realpath of dataset

if __name__ == '__main__':
    '''construct_labels(data_dir='/home/hume-users/leebri2n/PROJECTS/dote_1070-1083/.datasets/rareplanes/rareplanes_nsi_test')
    exit()
    print("Testing")'''
    yolo_dataset('test_dataset', data_config_dir='/home/hume-users/leebri2n/PROJECTS/dote_1070-1083/codex_te',
                 data_dir='/home/hume-users/leebri2n/PROJECTS/dote_1070-1083/.datasets/rareplanes/rareplanes_nsi_testing',
                 train_ids=['1_104005000FDC8D00_tile_8', '1_104005000FDC8D00_tile_13', '1_104005000FDC8D00_tile_14'],
                 val_ids=['1_104005000FDC8D00_tile_28'],
                 test_ids=['1_104005000FDC8D00_tile_29'])