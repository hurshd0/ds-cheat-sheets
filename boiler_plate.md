```python
#################### BOILER PLATE ###################
%matplotlib inline

import numpy as np # Linear algebra lib
import pandas as pd # Data analysis lib
import matplotlib.pyplot as plt # plotting lib
import seaborn as sns # matplotlib wrapper plotting lib
import random # python random lib
import pymc3 as pm # statmodelling lib
import theano.tensor as tt 
import tensorflow as tf

# Import data structures from collections
from collections import OrderedDict

# Matplotlib and Seaborn params
from matplotlib import rcParams
rcParams['figure.figsize'] = 10, 6
# Pick style of plots 
# Different style sheets: https://matplotlib.org/3.1.0/gallery/style_sheets/style_sheets_reference.html
plt.style.use('seaborn-darkgrid')
sns.set_context('notebook')

# Removes rows and columns truncation of '...'
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Ignore the warnings
import warnings
warnings.filterwarnings("ignore")

# Set folder path --- Optional
#import sys
#sys.path.append('')

# Add Scipy Stats
# https://docs.scipy.org/doc/scipy/reference/stats.html
from scipy.stats import stats

def check_versions():
    print('Numpy v{}'.format(np.__version__))
    print('Pandas v{}'.format(pd.__version__))
    print('Matplotlib v{}'.format(mpl.__version__))
    print('Seaborn v{}'.format(sns.__version__))
    print('PyMC3 v{}'.format(pm.__version__))

check_versions()
#######################################################
```
