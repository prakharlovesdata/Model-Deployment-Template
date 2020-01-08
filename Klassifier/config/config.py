import os
import pathlib

import pandas as pd

import Klassifier

PACKAGE_ROOT = pathlib.Path(Klassifier.__file__).resolve().parent



TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

# data
TRAINING_DATA_FILE = 'train.csv'
TESTING_DATA_FILE = 'test.csv'
TARGET = 'source'

FEATURES = ['num1', 'num2', 'num3',
            'num4', 'num5', 'num6',
            'num7', 'num8', 'num9', 'num10']

PIPELINE_NAME = 'knn_classification'
PIPELINE_SAVE_FILE = f'{PIPELINE_NAME}_output_v'

# used for differential testing
ACCEPTABLE_MODEL_DIFFERENCE = 0.05

# with open(os.path.join(PACKAGE_ROOT, 'VERSION')) as version_file:
#     __version__ = version_file.read().strip()
