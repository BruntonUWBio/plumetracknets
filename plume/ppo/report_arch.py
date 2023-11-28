#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
jupyter-nbconvert report_arch.ipynb --to python
python -u report_arch.py --expt_dir ~/plume/plumezoo/latest/fly/memory/
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
import pandas as pd

import numpy as np
import glob
import sys
sys.path.append('../')
import agent_analysis
import os
import importlib
import log_analysis
importlib.reload(log_analysis)
import torch
import arch_utils as archu
from statannot import add_stat_annotation # https://stackoverflow.com/questions/36578458/how-does-one-insert-statistical-annotations-stars-or-p-values-into-matplotlib/37518947#37518947



# In[3]:


import sys
batchmode = False
if 'ipykernel_launcher' in sys.argv[0]:
    print("Interactive mode")
else:
    batchmode = True
    print("Batch/CLI mode")
    import argparse


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
# print(plt.style.available)

mpl.rcParams['figure.dpi'] = 100
dpi_save = 300
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 10}
matplotlib.rc('font', **font)
mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']

# matplotlib.rc('font', **{'family': 'sans-serif', 
#                          'sans-serif': 'Arial', 
#                          'weight': 'normal', 
#                          'size': 9})

# Interactive vs. CLI
if not batchmode:
    get_ipython().run_line_magic('matplotlib', 'inline')
    # %config InlineBackend.figure_format = 'svg'
    get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
else:
    matplotlib.use("Agg")
    


# In[5]:


if batchmode:
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_dir', default=None)
    args = parser.parse_args()
    print(args)
    
    model_dir = args.model_dir
else:
    model_dir = natsorted(glob.glob(f'/home/satsingh/plume/plumezoo/latest/fly/memory/*VRNN*/'))[0]
#     model_dir = natsorted(glob.glob(f'/home/satsingh/plume/plumezoo/latest/fly/memory/*VRNN*/'))[-2]

outdir = f'{model_dir}/report_arch/'
os.makedirs(outdir, exist_ok=True)


# In[6]:


model_fname = model_dir[:-1] + ".pt"
model_fname
# models = natsorted(glob.glob(f'{expt_dir}/*VRNN*.pt'))
# assert len(models) > 0
# models
# models = ['/home/satsingh/plume/plumezoo/20210506/fly_all/memory/plume_20210418_VRNN_constantx5b5noisy6x5b5_bx1.0_t1M_w3_stepoob_h64_wd0.01_codeVRNN_seed19507d3.pt']


# In[7]:


model_dirs = natsorted(glob.glob(f'/home/satsingh/plume/plumezoo/latest/fly/memory/*VRNN*/'))
[x.split('seed')[-1] for x in model_dirs]


# # Architecture: Connectivity matrix
# Seems to only work for Vanilla RNN

# In[ ]:





# In[8]:


importlib.reload(log_analysis)
# for model_fname in models:

model_seed = model_fname.split('seed')[-1].split('.')[0]
figsize1=(4,3.5)
figsize1=(5,4)
figsize2=(4,3.5)
#     figsize1=(config.mwidth/2, 2)
#     figsize2=(config.mwidth/2, 2)



# ---- Plot 1: Eigenspectra ---- #    
fig, axs = plt.subplots(nrows=1, ncols=2, 
                    figsize=figsize1,  
                    sharey=True, 
                        sharex=True)
try:
    J0 = archu.get_J(model_fname + '.start') # Before training
    eig_vals, eig_vecs = archu.plot_eig(J0, axs[0], title=f'Before training', dotcolor='darkorange')    
except Exception as e:
    print(e)
J = archu.get_J(model_fname) # After training
eig_vals, eig_vecs = archu.plot_eig(J, axs[1], title='After training', dotcolor='royalblue')    
axs[0].set_xlabel('Real')
axs[1].set_xlabel('Real')
axs[0].set_ylabel('Imaginary')

plt.tight_layout()
# plt.suptitle(r'Eigenvalues of \mathbf{W}_h')
fname = f"{outdir}/eigenspectra_{model_seed}.png"
plt.savefig(fname, dpi=dpi_save, bbox_inches='tight')
plt.show()
print("Saved:", fname)

# ---- Plot 2: Timescales ---- #
# Old old
#     fig, axs = plt.subplots(nrows=1, ncols=2, 
#                         figsize=figsize2,  
#                         sharey=True, 
#                             sharex=True)
#     taus_J0 = archu.get_taus(J0)
#     taus_J = archu.get_taus(J)
#     archu.plot_all_taus(taus_J0, ax=axs[0], title=f'Before [{model_seed}]')
#     archu.plot_all_taus(taus_J, ax=axs[1], title="After")
#     plt.ylim(0, 2000)
#     plt.tight_layout()
#     fname = f"{outdir}/timescales_{model_seed}.png"
#     plt.savefig(fname, dpi=dpi_save, bbox_inches='tight')
#     plt.show()
#     print("Saved:", fname)
#     periods = archu.get_periods(eig_vals)
#     print(f"Periods [{model_seed}]: {periods}")

# Old - separate plot
figsize2=(3,2.5)
plt.figure(figsize=figsize2)
taus_J0 = archu.get_taus(J0)
taus_J = archu.get_taus(J)
pd.Series(taus_J0).plot(label='Before training', c='darkorange')
pd.Series(taus_J).plot(label='After training', c='royalblue')
ax = plt.gca()
ax.set_yscale("log")
plt.ylim(0, 2000)
plt.xlabel('Mode index')
plt.ylabel('Timesteps')
plt.tight_layout()

ax.set_yticks([12, 300])
ax.set_yticklabels([12, 300])
ax.axhline(12, c='grey', linestyle=':')
ax.axhline(300, c='grey', linestyle=':')

# ax.axhline(12, c='grey', linestyle=':', label=r"$\tau$=12")
# ax.axhline(300, c='grey', linestyle='dashed', label=r"$\tau$=300")

plt.legend(loc='upper right', labelspacing=0.07, bbox_to_anchor=(1.02, 1.02))
fname = f"{outdir}/timescales_{model_seed}.png"
plt.savefig(fname, dpi=dpi_save, bbox_inches='tight')
plt.show()
print("Saved:", fname)


# plt.subplots_adjust(top=0.95) # https://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots-in-matplotlib


# In[9]:


# eig_vals, eig_vecs = np.linalg.eig(J)
# eig_vals[ np.imag(eig_vals) != 0 ]


# In[ ]:





# # Biclustering

# In[10]:


# # https://scikit-learn.org/stable/auto_examples/bicluster/plot_spectral_biclustering.html#sphx-glr-auto-examples-bicluster-plot-spectral-biclustering-py
# # TODO: Not sure how to evaluate
# from sklearn.cluster import SpectralBiclustering
# from sklearn.metrics import consensus_score

# n_clusters = 4
# model = SpectralBiclustering(n_clusters=n_clusters, 
#                              method='log',
#                              random_state=0)

# model.fit(J)
# # plt.matshow(model.biclusters_[0], cmap='RdBu')


# In[11]:


# fit_data = J[np.argsort(model.row_labels_)]
# fit_data = fit_data[:, np.argsort(model.column_labels_)]

# plt.matshow(fit_data, cmap='RdBu')
# plt.colorbar()


# # Eigenvalue dynamics

# In[12]:


# # Random initialization Normal[0, 1/sqrt(N)]
# J = np.random.normal(size=(64, 64), loc=0, scale=1./np.sqrt(64))
# # J = np.random.normal(size=(64, 64))
# # plot_J(J)
# plot_eig(J)    
# # plot_hist(J)


# In[13]:


# fig, axs = plt.subplots(nrows=1, ncols=2, 
# #                             constrained_layout=True,
#                     figsize=(5,3),  
#                     sharey=True, 
#                         sharex=True)

# # Before training
# try:
#     J0 = archu.get_J(model_fname + '.start')
#     archu.plot_eig(J0, axs[0], title='Before')    
# #         plot_J(J)
# #         plot_hist(J)
# except Exception as e:
#     print(e)

# # After training
# J = archu.get_J(model_fname)
# archu.plot_eig(J, axs[1], title='After')    
# #     plot_J(J)
# #     plot_hist(J)
# taus = archu.get_taus(J)

# # Common 
# seed = model_fname.split('seed')[-1].split('.')[0]
# #     fig.suptitle(f"[ID:{model_seed}] Eigenspectra + Timescales {taus} ")
# plt.tight_layout()
# #     plt.subplots_adjust(top=0.95) # https://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots-in-matplotlib
# fname = f"{outdir}/eigenspectra_{model_seed}.png"
# plt.savefig(fname, dpi=dpi_save)
# plt.show()
# print("Saved:", fname)

# # Timescales
# fig, axs = plt.subplots(nrows=1, ncols=2, 
#                     figsize=(5,2.5),  
#                     sharey=True, 
#                         sharex=True)
# archu.plot_all_taus(J0, ax=axs[0], title="Before")
# archu.plot_all_taus(J, ax=axs[1], title="After")
# plt.show()
# plt.tight_layout()
# #     plt.subplots_adjust(top=0.95) # https://stackoverflow.com/questions/6541123/improve-subplot-size-spacing-with-many-subplots-in-matplotlib
# fname = f"{outdir}/timescales_{model_seed}.png"
# plt.savefig(fname, dpi=dpi_save)
# plt.show()
# print("Saved:", fname)


# In[14]:


eig_vals, eig_vecs = np.linalg.eig(J)
# eig_vals[ np.imag(eig_vals) != 0 ]
eig_vals[ np.linalg.norm(eig_vals) > 1 ]

ev = 1.0121603 +0.28471j
angular_velocity_degrees = np.angle(ev)*180/np.pi
period = 360/angular_velocity_degrees
period = 2*np.pi/np.angle(ev)
angular_velocity_degrees, period


# ## Jacobian Animation

# In[15]:


actor_critic, ob_rms = \
        torch.load(model_fname, map_location=torch.device('cpu'))
net = actor_critic.base.rnn #.weight_hh_l0.detach().numpy()
J0 = net.weight_hh_l0.detach().numpy()
archu.plot_eig(J0)    


# In[16]:


# J1 = Jx[1].squeeze().numpy()
# J2 = Jh[1].squeeze().numpy()
Jx, Jh = archu.get_jacobians(net, h=None, x=None)
eig_vals, eig_vecs = archu.plot_eig(Jh)

unstable_power, total_power, upr = archu.get_powers(eig_vals)
plt.title(f"UPR: {upr:.2f}")


# In[ ]:





# In[17]:


import arch_utils as archu
importlib.reload(archu)

# Save video for one trajectory
model_dir = model_fname.replace('.pt', '/')
is_recurrent = True if ('GRU' in model_dir) or ('VRNN' in model_dir) else False
log_fname = natsorted(glob.glob(model_dir + '*.pkl'))[0]
log_fname
with open(log_fname, 'rb') as f_handle:
    episode_logs = pickle.load(f_handle)

# Sequentially process trajectory
# ep_idx = 0
# log = episode_logs[ep_idx]
# ep_activity = log_analysis.get_activity(log, is_recurrent, do_plot=False)

for idx in range(len(episode_logs)):
    log = episode_logs[idx]
    outcome = log['infos'][-1][0]['done']
#     if outcome != 'HOME':
#         break
    if outcome == 'HOME':
        break

agent_analysis.visualize_episodes([log], 
                                  zoom=2, 
                                  dataset='constantx5b5',
                                  animate=False,
                                 )


# In[18]:


eig_df = archu.get_eig_df_episode(net, log)
eig_df.head()

traj_df = log_analysis.get_traj_df(log, 
                               extended_metadata=False, 
                               squash_action=True)
eig_df['odor_01'] = traj_df['odor_01']

    
# archu.animate_Jh_episode(eig_df, fname_suffix='test', outprefix="./")


# In[19]:


eig_df.columns
col_subset = ['unstable_power', 'stable_power', 'total_power', 'upr', 'odor_01']
eig_df.loc[:,col_subset].plot(subplots=True)


# ### Back to manuscript -- generate taus, eigs etc. over multiple episodes

# In[20]:


# 1 episode of each type
use_datasets = ['constantx5b5', 'switch45x5b5', 'noisy3x5b5']
selected_df = log_analysis.get_selected_df(model_dir, 
                              use_datasets, 
                              n_episodes_home=1, 
                              n_episodes_other=1,
                              min_ep_steps=0)

total_eps = len(selected_df['idx'].unique())
total_steps = selected_df['ep_length'].sum()
print(f"total_steps: {total_steps}, total_eps: {total_eps}")
selected_df


# In[21]:


taus_episodes = []
eig_dfs_all = []
traj_dfs_all = [] 
squash_action = True
for idx, row in selected_df.iterrows():
    log = row['log']
    eig_df = archu.get_eig_df_episode(net, log)
    eig_dfs_all.append( eig_df )
    
    taus_episode = pd.DataFrame(eig_df['timescales'].to_list())
    taus_episodes.append(taus_episode)

    traj_df = log_analysis.get_traj_df(log, 
                                   extended_metadata=False, 
                                   squash_action=squash_action)
    traj_dfs_all.append( traj_df )

taus_episodes = pd.concat(taus_episodes).reset_index(drop=True)
eig_dfs_all = pd.concat(eig_dfs_all).reset_index(drop=True)
traj_dfs_all = pd.concat(traj_dfs_all).reset_index(drop=True)


# In[22]:


taus_episodes.mean()
# yerrs = taus_episodes.std()
# taus_episodes.mean().plot()
# taus_episodes.std().plot(logy=True) # barely visible in log-axis
top5tau = taus_episodes.mean().head().to_list()
top5tau = np.around(top5tau, decimals=1)
print(f"{model_seed}: top5taus -- {top5tau}")


# In[23]:


# Timescales plot with CI
figsize2=(3,2.5)
plt.figure(figsize=figsize2)

# taus_J0 = archu.get_taus(J0) # use the one generated at the very top

g = sns.lineplot(data=taus_episodes.melt(), 
                 x='variable', 
                 y='value', 
                 color='royalblue',
#                  label=f'After training', # main manuscript
                 label=f'After training\n[N={total_steps} E={total_eps}]', # appendices
#                  estimator=np.median,
#                  estimator=np.mean,
                 ci=99,
                )
g.set_yscale("log")

pd.Series(taus_J0).plot(label='Before training', color='darkorange')
ax = plt.gca()

ax.set_yticks([12, 100, 300, 1000])
ax.set_yticklabels([12, 100, 300, 1000])
ax.tick_params(axis='y', which='major', left=True, length=3, width=1)
ax.axhline(12, c='grey', linestyle=':')
ax.axhline(300, c='grey', linestyle=':')


# ax.axhline(12, c='grey', linestyle=':', label=r"$\tau$=12")
# ax.axhline(300, c='grey', linestyle='dashed', label=r"$\tau$=300")

plt.ylim(0, 2000)
plt.xlim(-2, 70)
plt.xlabel('Mode index')
plt.ylabel('Timesteps')
plt.tight_layout()

plt.legend(loc='upper right', labelspacing=0.07, bbox_to_anchor=(1.02, 1.02))
fname = f"{outdir}/timescales_{model_seed}.png"
plt.savefig(fname, dpi=dpi_save, bbox_inches='tight')
plt.show()
print("Saved:", fname)


# In[24]:


eig_dfs_all['odor_01'] = traj_dfs_all['odor_01'].to_list()
eig_dfs_all['odor_01'] = eig_dfs_all['odor_01'].apply( lambda x : 'off' if x == 0 else 'on')
eig_dfs_all.head()
eig_dfs_all.tail()
# eig_dfs_all.reset_index(drop=True, inplace=True)


# In[25]:


# Max eigenvalue vs. odor on/off
# remove grid: https://stackoverflow.com/questions/26868304/how-to-get-rid-of-grid-lines-when-plotting-with-seaborn-pandas-with-secondary
# sns.set_style("white", {'axes.grid' : False})
eig_dfs_all['eig_max'] = eig_dfs_all['eig_vals'].apply( lambda ev : np.max(np.absolute(ev)) )

# axes = eig_dfs_all.boxplot(column='eig_max', by='odor_01', figsize=figsize2, grid=False)
ax, fig = plt.subplots(figsize=figsize2)
axes = sns.boxplot(data=eig_dfs_all, x='odor_01', y='eig_max')
add_stat_annotation(axes, data=eig_dfs_all, x='odor_01', y='eig_max',
                    box_pairs=[('on','off')],
                    test='Mann-Whitney', 
                    text_format='star', 
                    loc='inside',
                    verbose=2, 
                   )

plt.title('')
plt.suptitle('')
plt.ylabel("Maximum abs. eigenvalue")
plt.xlabel("Odor")
ax = plt.gca()
# ax.set_xticklabels(['off', 'on'])

# plt.legend(loc='upper right', labelspacing=0.07, bbox_to_anchor=(1.02, 1.02))

fname = f"{outdir}/eigmax_{model_seed}.png"
plt.savefig(fname, dpi=dpi_save, bbox_inches='tight')
plt.show()
print("Saved:", fname)


# In[26]:


# Unstable eigenvalue power vs. odor on/off

# remove grid: https://stackoverflow.com/questions/26868304/how-to-get-rid-of-grid-lines-when-plotting-with-seaborn-pandas-with-secondary
# sns.set_style("white", {'axes.grid' : False})
# eig_dfs_all['n_unstable'] = eig_dfs_all['eig_vals'].apply( lambda ev : np.max(np.absolute(ev)) )

# axes = eig_dfs_all.boxplot(column='eig_max', by='odor_01', figsize=figsize2, grid=False)
ax, fig = plt.subplots(figsize=(2.75,2.25))
axes = sns.boxplot(data=eig_dfs_all, x='odor_01', y='unstable_power')
add_stat_annotation(axes, data=eig_dfs_all, x='odor_01', y='unstable_power',
                    box_pairs=[('on','off')],
                    test='Mann-Whitney', 
                    text_format='star', 
                    loc='inside',
                    verbose=2, 
                   )

plt.title('')
plt.suptitle('')
plt.ylabel("Unstable eig. power")
plt.xlabel("Odor")
ax = plt.gca()
ax.set_xticklabels(['On', 'Off'])


# # plt.legend(loc='upper right', labelspacing=0.07, bbox_to_anchor=(1.02, 1.02))

fname = f"{outdir}/eigpower_{model_seed}.png"
plt.savefig(fname, dpi=dpi_save, bbox_inches='tight')
plt.show()
print("Saved:", fname)

figsize1, figsize2


# In[27]:


# ax, fig = plt.subplots(figsize=figsize2)
sns.displot(eig_dfs_all, x="unstable_power", hue="odor_01", height=figsize2[0])


# In[28]:


# import matplotlib.pyplot as plt
# import seaborn as sns
# from statannot import add_stat_annotation
# https://stackoverflow.com/questions/36578458/how-does-one-insert-statistical-annotations-stars-or-p-values-into-matplotlib/37518947#37518947
# sns.set(style="whitegrid")
df = sns.load_dataset("tips")

x = "size"
y = "total_bill"
ax = sns.boxplot(data=df, x=x, y=y)
add_stat_annotation(ax, data=df, x=x, y=y,
                    box_pairs=[(2, 3)],
#                     box_pairs=[("Thur", "Fri"), ("Thur", "Sat"), ("Fri", "Sun")],
                    test='Mann-Whitney', text_format='star', loc='inside', verbose=2)

df


# In[29]:


# plot_columns = ['total_power', 'stable_power', 'unstable_power', 'upr']
# axs = eig_df.loc[:,plot_columns].plot(subplots=True, figsize=(10, 4))
# for ax in axs:
#     ax.legend(loc='lower left')
# eig_df.columns


# In[30]:


# # Eigenvector projections
ep_activity = log_analysis.get_activity(log, is_recurrent=True, do_plot=False)
# J0 = net.weight_hh_l0.detach().numpy()
eig_vals, eig_vecs = archu.plot_eig(J0)

# Jx, Jh = archu.get_jacobians(net, h=None, x=None)
# eig_vals, eig_vecs = archu.plot_eig(Jh)

# eligible_idxs = np.absolute(eig_vals) > 1.0
# evecs_unstable = eig_vecs[ eligible_idxs ]
# evecs_unstable.shape, ep_activity.shape

# projections = evecs_unstable @ ep_activity.T
# proj_df = np.absolute(pd.DataFrame(projections.T))
# proj_df.columns = [str(np.around(x, decimals=2)) for x in eig_vals[eligible_idxs] ]
# axs = proj_df.plot(subplots=True, figsize=(12, 6))

# for i in range(len(axs)):
#     axs[i].set_label(['A'])
#     axs[i].legend(loc='lower left')
    
# plt.savefig()
# # plt.titles("Neural activity projected onto ")
# # np.absolute(proj_df.head())

archu.plot_eigvec_projections(eig_vals, eig_vecs, ep_activity, 
    fname_suffix='test', outprefix="./")


# In[31]:


squash_action = True
traj_df = log_analysis.get_traj_df(log, 
                extended_metadata=True, squash_action=squash_action)

print(traj_df.columns)
plot_cols = ['wind_theta_obs', 
#              'wind_x_obs', 
#              'wind_y_obs', 
             'step', 
             'turn',
             'odor_obs',
             'stray_distance', 
#              'odor_01', 
             'odor_clip',
             'odor_lastenc',
#              'wind_speed_ground',
#              'wind_angle_ground',
             'agent_angle_ground',
             'odor_ma_8',
             'r_step',
            ]

traj_df.loc[:, plot_cols].plot(subplots=True, figsize=(8,6))


# In[32]:


# # Choose random subset of RNN states from all HOME/!HOME states
# model_dir = model_fname.replace('.pt', '/')
# is_recurrent = True if ('GRU' in model_dir) or ('VRNN' in model_dir) else False
# log_fname = natsorted(glob.glob(model_dir + '*.pkl'))[0]
# log_fname
# with open(log_fname, 'rb') as f_handle:
#     episode_logs = pickle.load(f_handle)
# h_episodes = []
# for idx in range(len(episode_logs)):
#     log = episode_logs[idx]
#     outcome = log['infos'][-1][0]['done']
# #     if outcome == 'HOME':
#     if outcome != 'HOME':
#         ep_activity = log_analysis.get_activity(log, is_recurrent, do_plot=False)
#         h_episodes.append(ep_activity)
    
# h_episodes_stacked = np.vstack(h_episodes)
# h_episodes_stacked.shape

# # select random subset
# max_hxs = 500
# max_hxs = min(max_hxs, h_episodes_stacked.shape[0])
# h_idxs = np.random.choice(h_episodes_stacked.shape[0], size=max_hxs, replace=False).astype(int)
# Jhs = []
# periods_list = []
# timescales_list = []
# for h_idx in tqdm.tqdm(h_idxs):
#     h = torch.tensor(h_episodes_stacked[h_idx,:].reshape(1, 1, -1), requires_grad=True)    
#     Jx, Jh = get_jacobians(net, h=h, x=None)
#     Jhs.append( Jh )
    
#     timescales_list.append( get_taus(Jh) )
#     eig_vals, eig_vecs = np.linalg.eig(Jh)
#     periods_list.append( get_periods(eig_vals) )
# #     plot_eig(Jh)
# len(Jhs)


# In[33]:


# Jhs
# periods_list
# timescales_list
# periods_list_flat = np.concatenate(flatten(periods_list))
# pd.Series(periods_list_flat).hist(bins=100)
# plt.xlim(0,50)


# In[34]:


# np.histogram(periods_list_flat)


# In[35]:


# flatten(timescales_list)


# In[ ]:





# In[ ]:




