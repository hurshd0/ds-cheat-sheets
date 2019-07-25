```python
#################### BOILER PLATE ###################
%matplotlib inline

import numpy as np # Linear algebra lib
import pandas as pd # Data analysis lib
import matplotlib.pyplot as plt # plotting lib
import seaborn as sns # matplotlib wrapper plotting lib
import random # python random lib

# Matplotlib and Seaborn params
from matplotlib import rcParams
rcParams['figure.figsize'] = 10, 6
# Pick style of plots 
# Different style sheets: https://matplotlib.org/3.1.0/gallery/style_sheets/style_sheets_reference.html
plt.style.use('ggplot')

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

#######################################################
```
