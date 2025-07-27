# Ignore Warnings
import warnings
import dill

# Data Manipulation
import pandas as pd
import numpy as np

# Imputation - RandomForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Data Transformation
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.preprocessing import StandardScaler

# Feature Selection
from sklearn.model_selection import train_test_split, GridSearchCV

# Pipeline
from sklearn.pipeline import Pipeline

# Metrics
import math
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.metrics import r2_score

# Regression Algorithms
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Disable warning outputs
warnings.filterwarnings('ignore')  

# Path to the CSV data file
TRAIN_DATAPATH = 'dynamic_pricing.csv' 

# Reading data from the CSV file into a DataFrame
df = pd.read_csv(TRAIN_DATAPATH)  

# Displaying the first few rows of the DataFrame
print(df.head())

# Initialize empty lists to store object and non-object columns
obj = []
ints = []

# Loop through DataFrame columns
for col in df.columns:
    if df[col].dtype == 'object':
        obj.append((col, df[col].nunique(), df[col].isna().sum()))
    else:
        ints.append((col, df[col].nunique(), df[col].isna().sum(), df[col].skew()))

# Make lengths equal
max_len = max(len(obj), len(ints))
obj.extend([('', '', '')] * (max_len - len(obj)))
ints.extend([('', '', '', '')] * (max_len - len(ints)))

# Construct data dictionary
data = {
    'Categorical_columns': [x[0] for x in obj],
    'cat_cols_uniques': [x[1] for x in obj],
    'cat_cols_missing': [x[2] for x in obj],
    'Numeric_columns': [x[0] for x in ints],
    'int_cols_uniques': [x[1] for x in ints],
    'int_cols_missing': [x[2] for x in ints],
    'int_cols_skew': [x[3] for x in ints]
}

# Display column-wise analysis
print(pd.DataFrame(data))
