```python
########################################### BOILER PLATE IMPORTS #############################################################

# Below line is necessary to display plots in notebook
%matplotlib inline

# Linear Algebra libs
import numpy as np # LA lib
import pandas as pd # Data analysis lib

# ML libs
# import theano.tensor as tt 
# import tensorflow as tf

# Plotting libs
import matplotlib.pyplot as plt # plotting lib
import seaborn as sns # matplotlib wrapper plotting lib
import plotly.graph_objs as go # interactive low-level plotting lib https://plot.ly/python/
import plotly_express as px #high-level api wrapper for plotly https://plot.ly/python/plotly-express/#visualize-distributions

# Stat libs
import random # https://docs.python.org/3.6/library/random.html
import pymc3 as pm # https://docs.pymc.io/nb_examples/index.html
from scipy.stats import stats # https://docs.scipy.org/doc/scipy/reference/stats.html

# DATA ANALYSIS
import pandas_profiling


# Python libs and other necessary libs
# import os
# import sys

# Set folder path --- Optional
# sys.path.append('')

# Import data structures from collections
# from collections import OrderedDict

# --- Matplotlib and Seaborn params --- #
# Pick style of Matplolib plots 

# Different style sheets:-> https://matplotlib.org/3.1.0/gallery/style_sheets/style_sheets_reference.html

# Configure Seaborn Asthetics: -> https://seaborn.pydata.org/tutorial/aesthetics.html?highlight=style
plt.style.use('seaborn-darkgrid')
sns.set(context='notebook', style='darkgrid', palette='colorblind')

# --- Pandas settings --- #
# Removes rows and columns truncation of '...'
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Ignore the warnings
import warnings
warnings.filterwarnings("ignore")


# Helper functions
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

# check_versions()


##############################################################################################################################
```
