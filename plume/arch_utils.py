from __future__ import division
import os
# import glob
# import pickle
from natsort import natsorted
# import argparse
import os
# import sys
import numpy as np
import tqdm
import pandas as pd

import numpy as np
from pprint import pprint
import glob
import sys
import config
# import agents
import agent_analysis
import os
# import sklearn
# import sklearn.decomposition as skld

# import importlib
import log_analysis
import torch

import contextlib
from moviepy.editor import ImageClip, concatenate_videoclips


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
        'size'   : 18}
matplotlib.rc('font', **font)
matplotlib.use("Agg")


def get_J(model_fname):
    actor_critic, ob_rms = \
            torch.load(model_fname, map_location=torch.device('cpu'))
#     actor_critic.base.rnn
    # dir(actor_critic.base.rnn)
    net = actor_critic.base.rnn #.weight_hh_l0.detach().numpy()
    # weights_hh
    # plt.matshow(weights_hh)
    # https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
    if 'GRU' in model_fname:
        w_ir, w_ii, w_in = net.weight_ih_l0.chunk(3, 0)
        print(w_ir.shape, w_ii.shape, w_in.shape)
        w_hr, w_hi, w_hn = net.weight_hh_l0.chunk(3, 0)
        print(w_hr.shape, w_hi.shape, w_hn.shape)
    #     J = w_hr.detach().numpy()
    #     J = w_hi.detach().numpy()
        J = w_hn.detach().numpy()
    if 'VRNN' in model_fname:
        J = net.weight_hh_l0.detach().numpy()
    return J

def plot_J(J, ax=None):
    if ax is None:
        fig, ax = plt.subplots()    
    ax.matshow(J, cmap='RdBu')
    ax.colorbar()
    if ax is None:
        plt.show()

def plot_hist(J):
    pd.Series(J.flatten()).hist(bins=50)
#     plt.yscale('log')
    plt.show()


def plot_eig(J, ax=None, title='', dotcolor='blue'):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
        
    eig_vals, eig_vecs = np.linalg.eig(J)
    circ = plt.Circle((0, 0), radius=1, edgecolor='red', facecolor='None', linestyle='dashed')
    ax.add_patch(circ)
    ax.axhline(0, c='grey', linestyle='dashed')
    ax.axvline(0, c='grey', linestyle='dashed')
    ax.set_aspect('equal', adjustable='box')
    ax.scatter(eig_vals.real, eig_vals.imag, c=dotcolor, edgecolor='None', s=15)
    ax.set_title(f'{title}')

    if ax is None:
        plt.show()
        
    return eig_vals, eig_vecs


def get_taus(J, filters=['fix']):
    # Use only eigenvalues with |\lambda| < 1 (i.e. decaying)
    eig_vals, eig_vecs = np.linalg.eig(J)
    eligible = eig_vals
    if 'stable' in filters:
        eligible = eligible[ np.absolute(eig_vals) < 1.0 ]
    if 'munstable' in filters:
        eligible = eligible[ np.absolute(eig_vals) >= 1.0 ] # unstable or marginally stable

    timescales = np.abs(1/np.log( np.absolute(eligible) ))
    if 'fix' in filters: # Fix marginally/unstable eig taus to 1
        timescales[ np.absolute(eig_vals) >= 1.0 ] = 1.0
    timescales = pd.Series(timescales).sort_values(ascending=False).tolist()    
    timescales = [np.around(t, 1) for t in timescales] # round to 0 decimals
    return timescales

def plot_all_taus(taus, ax=None, title=''):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    tauss = pd.Series(taus).sort_values(ascending=False).reset_index(drop=True)
    tauss.plot(logy=True, ax=ax, ylabel="Timesteps", xlabel="Index", label="_")
    ax.set_title(f'{title}')
    ax.axhline(12, c='grey', linestyle=':', label=r"$\tau$=12")
    ax.axhline(300, c='grey', linestyle='dashed', label=r"$\tau$=300")
    # ax.axhline(300, c='grey', linestyle='dashed', label='Max. ep. length')
    plt.legend(loc='upper right')
    if ax is None:
        plt.show()

def get_periods(eig_vals, max_period=300):
    periods = 2*np.pi/np.angle(eig_vals[ np.absolute(eig_vals) >= 1.0 ])
    periods = np.unique(np.abs(periods))
    periods = periods[periods <= max_period]
    #     plt.scatter(np.periods) 
    #     pd.Series(periods).sort_values(ascending=False).plot()
    #     plt.show()

    return periods





# TODO: https://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/
def get_jacobians(net, h=None, x=None):
#     J = net.weight_hh_l0.detach().numpy()
    if x is None:
        x = torch.zeros(1, 1, net.input_size, requires_grad=True)
    if h is None:
        h = torch.zeros(1, 1, net.hidden_size, requires_grad=True)
    Jx, Jh = torch.autograd.functional.jacobian(net, (x, h))
    
    # Jx[0].squeeze().numpy().shape # (64, 3)
    # Jx[1].squeeze().numpy().shape # (64, 64)
    # Jh[0].squeeze().numpy().shape # (64, 3)
    # Jh[1].squeeze().numpy().shape # (64, 64)

    return Jx[0].squeeze().numpy(), Jh[1].squeeze().numpy()


def get_powers(eig_vals):
    unstable_eigvals = eig_vals[ np.absolute(eig_vals) > 1.0 ] 
    stable_eigvals = eig_vals[ np.absolute(eig_vals) <= 1.0 ] 

    unstable_power = np.sum(np.power(np.absolute(unstable_eigvals), 2))
    total_power = np.sum(np.power(np.absolute(eig_vals), 2))
    upr = unstable_power/total_power
    return unstable_power, total_power, upr

# https://stackabuse.com/python-how-to-flatten-list-of-lists
def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


def get_periods(eig_vals, max_period=300, min_period=4):
    periods = np.pi/np.angle(eig_vals[ np.absolute(eig_vals) > 1.0 ])
#     periods = np.pi/np.angle(eig_vals)
    periods = np.unique(np.abs(periods))
    periods = periods[periods <= max_period]
    periods = periods[periods >= min_period]
    return periods


def get_eig_df_episode(net, log):
    ep_activity = log_analysis.get_activity(log, is_recurrent=True, do_plot=False)

    # Extract jacobians and associated eigenvalues
    eig_df = []
    for h_idx in tqdm.tqdm(range( ep_activity.shape[0] )):
        h = torch.tensor(ep_activity[h_idx,:].reshape(1, 1, -1), requires_grad=True)    
        Jx, Jh = get_jacobians(net, h=h, x=None)
        
        eig_vals, eig_vecs = np.linalg.eig(Jh)

        unstable_power, total_power, upr = get_powers(eig_vals)
        eig_df.append({
            'idx': h_idx,
            'unstable_power': unstable_power,
            'stable_power': total_power - unstable_power,
            'total_power': total_power,
            'upr': upr,
            'periods': get_periods(eig_vals),
            'timescales': get_taus(Jh),
            'eig_vals': eig_vals,
            'Jh': Jh,
        })

    # Plot eigenvalues
    eig_df = pd.DataFrame(eig_df)
    eig_df.head()

    return eig_df

def plot_eigvec_projections(eig_vals, eig_vecs, ep_activity, 
    fname_suffix='test', outprefix="./"):
    # Eigenvector projections
    eligible_idxs = np.absolute(eig_vals) > 1.0
    evecs_unstable = eig_vecs[ eligible_idxs ]
    evecs_unstable.shape, ep_activity.shape

    projections = evecs_unstable @ ep_activity.T
    proj_df = np.absolute(pd.DataFrame(projections.T))
    proj_df.columns = [str(np.around(x, decimals=2)) for x in eig_vals[eligible_idxs] ]

    axs = proj_df.plot(subplots=True, figsize=(10, 6))
    for i in range(len(axs)):
        axs[i].set_label(['A'])
        axs[i].legend(loc='lower left')        
    fname = f'{outprefix}/evprojections_{fname_suffix}.png'
    plt.savefig(fname)
    print("Saved:", fname)

def get_minmax_eigv(eig_df):
    min_x, max_x = -1., 1.
    min_y, max_y = -1., 1.

    for idx, row in eig_df.iterrows():
        eig_vals = row['eig_vals']
        min_x = min(min_x, np.min(np.real(eig_vals)))
        min_y = min(min_y, np.min(np.imag(eig_vals)))
        max_x = max(max_x, np.max(np.real(eig_vals)))
        max_y = max(max_y, np.max(np.imag(eig_vals)))

    return min_x, max_x, min_y, max_y

def animate_Jh_episode(eig_df, fname_suffix='test', outprefix="./"):
    min_x, max_x, min_y, max_y = get_minmax_eigv(eig_df)

    if not os.path.exists(f'{outprefix}/tmp/'):
        os.makedirs(f'{outprefix}/tmp/')

    output_fnames = []
    for idx, row in tqdm.tqdm(eig_df.iterrows()):
        fig, ax = plt.subplots(nrows=1, ncols=1, 
            figsize=(2.5,2.5), sharey=True, sharex=True)
        plt.xlim(min_x-0.1, max_x+0.1)
        plt.ylim(min_y-0.1, max_y+0.1)

        # title = f"{idx:03d} - UPR:{row['upr']:.2f} TP:{row['total_power']:.2f}"
        title = f"{idx:03d}"
        fname = f"{outprefix}/eigJ_{fname_suffix}_{idx:05d}.png"
        output_fnames.append(fname)
        plot_eig(row['Jh'], ax=ax, title=title)
        plt.savefig(fname)
        
    # Create video & delete frames
    output_fnames = natsorted(output_fnames,reverse=False)
    clips = [ImageClip(f).set_duration(0.08) for f in output_fnames] # 
    concat_clip = concatenate_videoclips(clips, method="compose")
    fanim = f'{outprefix}/{fname_suffix}_eig.mp4'
    concat_clip.write_videofile(fanim, fps=15, verbose=False, logger=None)    
    print("Saved:", fanim)

    # https://stackoverflow.com/questions/10840533/most-pythonic-way-to-delete-a-file-which-may-not-exist
    for f in output_fnames:
        with contextlib.suppress(FileNotFoundError):
            os.remove(f)

    # Save eigen-value power trajectory
    plot_columns = ['total_power', 'stable_power', 'unstable_power', 'upr']
    axs = eig_df.loc[:,plot_columns].plot(subplots=True)
    for ax in axs:
        ax.legend(loc='lower left')
    eig_df.columns
    fname = f'{outprefix}/eigpower_{fname_suffix}.png'
    plt.savefig(fname)
    print("Saved:", fname)