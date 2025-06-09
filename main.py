import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import datetime as dt

# Data Viz 
import seaborn as sns
import matplotlib.pyplot as plt

# Data Manipulation
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Similarity calculation
from sklearn.metrics.pairwise import cosine_similarity

# Import ML libraries
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# settings
pd.options.display.max_rows = 100
pd.options.display.max_columns = None

# Math
import math

# Remove warnings
import warnings
warnings.filterwarnings("ignore")

import os
curent_dir = os.getcwd()
dataset_path = os.path.join(curent_dir, 'dataset')
train_csv_path = os.path.join(dataset_path, 'train_ver2.csv.zip')
test_csv_path = os.path.join(dataset_path, 'test_ver2.csv.zip')

train = pd.read_csv(filepath_or_buffer=train_csv_path)
test = pd.read_csv(filepath_or_buffer=test_csv_path)

## Data Profiling
train.info()