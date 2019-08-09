```python
############################ BOILER PLATE IMPORTS ##############################

# Below line is necessary to display plots in notebook
################################ EDA IMPORTS ###################################
%matplotlib inline
import pandas as pd 
import pandas_profiling # Quick Data Analysis
import numpy as np # Linear Algebra lib
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.graph_objs as go # interactive low-level plotting lib https://plot.ly/python/
import plotly.express as px #high-level api wrapper for plotly https://plot.ly/python/plotly-express/#visualize-distributions
# ---------------- Plot libs settings ------------- #
# Pick style of Matplolib plots 
# Different style sheets:-> https://matplotlib.org/3.1.0/gallery/style_sheets/style_sheets_reference.html
# Configure Seaborn Asthetics: -> https://seaborn.pydata.org/tutorial/aesthetics.html?highlight=style
plt.style.use('seaborn-darkgrid')
sns.set(context='notebook', style='darkgrid', palette='colorblind')
# Seting a universal figure size 
plt.rcParams['figure.figsize'] = (10, 6)

# ---------------- Pandas settings --------------- #
# Removes rows and columns truncation of '...'
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
################################################################################

################################ STATS IMPORTS #################################
import random # https://docs.python.org/3.6/library/random.html
import pymc3 as pm # https://docs.pymc.io/nb_examples/index.html
from scipy.stats import stats # https://docs.scipy.org/doc/scipy/reference/stats.html
################################################################################

#################################### ML IMPORTS ################################
### Pre-Processing
import category_encoders as ce
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.impute import SimpleImputer


### Score Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap  # package used to calculate Shap values

### Train/Test - Split & CV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold


### Pipeline
from sklearn.pipeline import make_pipeline

### Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import xgboost as xgb

### Regression 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Hyperopt modules
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
from functools import partial
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

################################################################################

############################### DEEP ML IMPORTS ################################
import tensorflow as tf
import torch
from keras import models
from keras import layers
from keras import optimizers
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
################################################################################

############################### PYTHON IMPORTS #################################
# Python libs and other necessary libs
# import os
# import sys
# import re

# Set folder path --- Optional
# sys.path.append('')

# Import data structures from collections
# from collections import OrderedDict

################################################################################

############################# Ignore the warnings ##############################
import warnings
warnings.filterwarnings("ignore")
################################################################################

############################### Helper functions ################################

def set_seed(seed):
    """
    Sets random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    
def check_versions():
    print('Numpy v{}'.format(np.__version__))
    print('Pandas v{}'.format(pd.__version__))
    print('Matplotlib v{}'.format(mpl.__version__))
    print('Seaborn v{}'.format(sns.__version__))
    print('PyMC3 v{}'.format(pm.__version__))
        

##################################################################################
```
