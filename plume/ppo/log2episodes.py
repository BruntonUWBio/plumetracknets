#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
jupyter-nbconvert log2episodes.ipynb --to python

python -u log2episodes.py
"""


# In[2]:


from __future__ import division
import os
import glob
import pickle
from natsort import natsorted
import argparse
import os
import sys
import numpy as np
import tqdm
import pandas as pd

import numpy as np
# from pprint import pprint
import glob
import sys
sys.path.append('../')

# from plume_env import PlumeEnvironment, PlumeFrameStackEnvironment
# import config
# not used and not in requirements.txt
# import agents
# import agent_analysis
import os
# import sklearn
# import sklearn.decomposition as skld

import importlib
# sys.path.append('/src/smartFlies/code/')
import log_analysis # custom script. Need to be added  
importlib.reload(log_analysis)


# In[3]:


import sys
batchmode = False
if 'ipykernel_launcher' in sys.argv[0]:
    print("Interactive mode")
else:
    batchmode = True
    print("Batch/CLI mode")
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--logfile', default=None)
    args = parser.parse_args()
    print(args)


# In[4]:


# Common
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

from IPython.display import clear_output
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
sns.set(style="white")
print(plt.style.available)

mpl.rcParams['figure.dpi'] = 144
dpi_save = 144


font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)

# Interactive vs. CLI
if not batchmode:
    get_ipython().run_line_magic('matplotlib', 'inline')
    # %config InlineBackend.figure_format = 'svg'
    get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
else:
    matplotlib.use("Agg")


# ### Load

# In[5]:



if batchmode:
    logfile = args.logfile
    log_fname = os.path.abspath(logfile) # get full path
else:
    expt_dir = '/home/satsingh/plume/plumezoo/latest/fly/memory/'
    log_fname = natsorted(glob.glob(f"{expt_dir}/plume_*/*.pkl"))[0] # Pick one

log_fname
# outdir = f"{expt_dir}/report_diffs/"
# os.makedirs(outdir, exist_ok=True)
# print(outdir)


# In[6]:


is_recurrent = True if ('GRU' in log_fname) or ('VRNN' in log_fname) else False



with open(log_fname, 'rb') as f_handle:
    episode_logs = pickle.load(f_handle)


# In[13]:


squash_action = True
ep_metadata_df = pd.DataFrame([ 
    log_analysis.get_episode_metadata(log, squash_action=squash_action) for log in tqdm.tqdm(episode_logs) ])


# In[14]:


# ep_metadata_df['idx'] = np.arange(len(ep_metadata_df), dtype=int)
ep_metadata_df.insert(0, 'idx', np.arange(len(ep_metadata_df), dtype=int))


# In[15]:


# print(ep_metadata_df['start_y'].unique())
print(ep_metadata_df.columns)
ep_metadata_df.head()


# In[ ]:


fname = log_fname.replace(".pkl", "_episodes.csv")
ep_metadata_df.to_csv(fname, index=False)
print("Saved", fname)


# In[10]:


# Plot start and end locations 
plt.scatter(ep_metadata_df['start_x'], ep_metadata_df['start_y'], c='black', s=10)

colors = {'HOME':'green', 'OOB':'red', 'OOT':'blue'} 
ax = plt.scatter(ep_metadata_df['end_x'], ep_metadata_df['end_y'], 
            c=[colors[x] for x in ep_metadata_df['done']], alpha=0.85, s=10)

ax = plt.gca()
ax.set_aspect('equal')
plt.xlim(-0.5, 10.5)
plt.ylim(-3, 3)

plt.tight_layout()

img_fname = log_fname.replace(".pkl", "_start_end.png")
plt.savefig(img_fname, dpi=dpi_save, bbox_inches='tight')
plt.show()
print("Saved", img_fname)


# In[ ]:




