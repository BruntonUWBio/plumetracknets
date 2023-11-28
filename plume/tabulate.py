#!/usr/bin/env python
# coding: utf-8

# In[14]:


"""
jupyter-nbconvert tabulate.ipynb --to python 
ipython tabulate.py -- --base_dir ~/plume/plumezoo/
ipython tabulate.py -- --base_dir ~/plume/plume2/ppo/trained_models/
"""


# In[15]:


import pandas as pd
import glob 
from pathlib import Path
import tqdm
import argparse


# In[16]:


import sys
batchmode = False
if 'ipykernel_launcher' in sys.argv[0]:
    print("Interactive mode")
    
    BASE_DIR="~/plume/plumezoo/latest/fly/memory/"
#     BASE_DIR="~/plume/plume2/ppo/trained_models/"
#     BASE_DIR="~/plume/plumezoo/"
    
else:
    batchmode = True
    print("Batch/CLI mode")
    import argparse
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--base_dir', default='./')
    args = parser.parse_args()
    BASE_DIR=args.base_dir


# In[17]:


print(BASE_DIR)
# fnames = Path(BASE_DIR).rglob('*VRNN*/*_summary.csv')
# fnames = !find {BASE_DIR} -name "*VRNN*/*_summary.csv"
fnames = get_ipython().getoutput('find {BASE_DIR} -name "*_summary.csv"')
# fnames


# In[18]:


counts_df = []
for fname in tqdm.tqdm(fnames):
    s = pd.read_csv(fname) 
    dataset = str(fname).split('/')[-1].replace('_summary.csv','')
    row = {
        'dataset': dataset,
        'HOME': sum(s['reason'] == 'HOME'),
        'OOB': sum(s['reason'] == 'OOB'),
        'OOT': sum(s['reason'] == 'OOT'),
        'total': len(s['reason']),
        'seed': str(fname).split('seed')[-1].split('/')[0],
        'model_dir': str(fname).replace(f'{dataset}_summary.csv','').replace(BASE_DIR,''),
        'fname': str(fname)
    }
    counts_df.append(row)
    
counts_df = pd.DataFrame(counts_df)


# In[19]:


eligible_datasets = [
        'constantx5b5', 
        'switch15x5b5', 
        'switch30x5b5', 
        'switch45x5b5', 
        'noisy3x5b5', 
        'noisy6x5b5',
        'constantx5b5_0.8',
        'constantx5b5_0.6',
        'constantx5b5_0.4', 
        'constantx5b5_0.2',
]


# In[20]:


print(counts_df.shape)
# counts_df = counts_df.query("total == 240 and dataset in @eligible_datasets")
counts_df = counts_df.query("dataset in @eligible_datasets")
print(counts_df.shape)


# In[21]:


counts_df['dataset'].unique()


# In[22]:


pivot_df = counts_df.pivot(index='model_dir', columns='dataset', values='HOME').reset_index()
pivot_df['total'] = pivot_df.sum(axis=1, skipna=True)


# In[23]:


col_order = eligible_datasets = [
        'total',
        'constantx5b5', 
        'switch45x5b5', 
        'noisy3x5b5', 
        'noisy6x5b5',
        'switch30x5b5', 
        'switch15x5b5', 
        'constantx5b5_0.8',
        'constantx5b5_0.6',
        'constantx5b5_0.4', 
        'constantx5b5_0.2',
        'model_dir',
]
pivot_df = pivot_df[col_order]
pivot_df = pivot_df.sort_values(by='total', ascending=False)
pivot_df


# In[24]:


pivot_df.to_csv(f'{BASE_DIR}/tabulated.tsv', sep='\t', index=False)
pivot_df.to_csv(f'{BASE_DIR}/tabulated.csv', index=False)


# In[12]:


# import seaborn as sns
# sns.pairplot(pivot_df)


# In[ ]:




