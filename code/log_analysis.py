import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle 

import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns

from array2gif import write_gif
from moviepy.editor import ImageClip, concatenate_videoclips
from natsort import natsorted
import contextlib
import os
import tqdm

import sklearn.decomposition as skld
import config
np.random.seed(config.seed_global)

from natsort import natsorted
import contextlib
import os
import tqdm


ODOR_THRESHOLD = config.env['odor_threshold']

def wind_xy_to_theta(x, y):
    """
    Standard CCW notation, centered at 0
    Returns +1 is +180-deg, and -0.999 is -180-deg 
    wind_xy_to_theta(0,0) # 0.0
    wind_xy_to_theta(1,0) # 0.0
    wind_xy_to_theta(1,1) # 0.25
    wind_xy_to_theta(0,1) # 0.5
    wind_xy_to_theta(-1,1) # 0.75
    wind_xy_to_theta(-1,0) # 1.0
    wind_xy_to_theta(-1,-0.01) # -0.997 
    wind_xy_to_theta(-1,-1) # -0.75
    wind_xy_to_theta(0,-1) # -0.5
    wind_xy_to_theta(+1,-1) # -0.25
    """
    return np.angle( x + 1j*y, deg=False )/np.pi # note div by np.pi!

def shift_scale_theta(theta):
    """
    input: between -1 (-180-deg) and +1 (+180 deg)
    output: between 0 (-180-deg) and +1 (+180 deg)

    """
    return (theta+1)/2

# TODO: Stack Overflow citation missing!
def rle(inarray):
    """ run length encoding. Partial credit to R rle function. 
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.asarray(inarray)                  # force numpy
    n = len(ia)
    if n == 0: 
        return (None, None, None)
    else:
        y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions
        return(z, p, ia[i])


def rescale_col(series, vmax=None):
    s = series - series.min()
    if vmax is not None:
        s /= vmax
    else:
        s /= s.max()
    return s


def get_selected_df(model_dir, use_datasets, 
    n_episodes_home=240, 
    n_episodes_other=240, 
    min_ep_steps=0, 
    oob_only=True,
    balanced=True):
    episodes_df = []
    for dataset in use_datasets:
        log_fname = f'{model_dir}/{dataset}.pkl'
        with open(log_fname, 'rb') as f_handle:
            episode_logs = pickle.load(f_handle)

        for idx in range(len(episode_logs)):
            log = episode_logs[idx]
            episodes_df.append({'dataset': dataset, 
                                'idx': idx,
                                'ep_length': len(log['trajectory']),
                                'log': log,
                                'outcome': log['infos'][-1][0]['done'],
                               })

    episodes_df = pd.DataFrame(episodes_df).sort_values(by='ep_length', ascending=False)
    episodes_df = episodes_df.query("ep_length >= @min_ep_steps")
    if oob_only:
        selected_df = pd.concat([ 
            episodes_df.query('outcome == "HOME"').groupby('dataset').head(n_episodes_home),
            episodes_df.query('outcome == "OOB"').groupby('dataset').head(n_episodes_other),
            pd.DataFrame({})
        ]).reset_index(drop=True)
    else:
        selected_df = pd.concat([ 
            episodes_df.query('outcome == "HOME"').groupby('dataset').head(n_episodes_home),
            episodes_df.query('outcome != "HOME"').groupby('dataset').head(n_episodes_other),
            pd.DataFrame({})
        ]).reset_index(drop=True)


    if balanced:
        counts_df = selected_df.groupby(['dataset', 'outcome']).count()
        counts_df

        # Balance out HOME and OOB in selected_df
        balanced_df = []
        for dataset in use_datasets:
            min_count = counts_df.query("dataset == @dataset")['idx'].min()
            balanced_df.append( selected_df.query("dataset == @dataset").groupby('outcome').head(min_count) )

        balanced_df = pd.concat(balanced_df)
        balanced_df.groupby(['dataset', 'outcome']).count()
        selected_df = balanced_df

    return(selected_df)

def get_pca_common(selected_df, n_comp = 12, is_recurrent=True):
    h_episodes = []
    traj_dfs = []
    # squash_action = True

    for episode_log in selected_df['log']:
        ep_activity = get_activity(episode_log, is_recurrent, do_plot=False)
        h_episodes.append(ep_activity)

    h_episodes_stacked = np.vstack(h_episodes)
    # print(h_episodes_stacked.shape)

    pca_common = skld.PCA(n_comp, whiten=False)
    # pca_common = skld.PCA(whiten=False)
    pca_common.fit(h_episodes_stacked)
    return pca_common

def regime_to_colors(regime_list):
    colors = [ config.regime_colormap[x] for x in regime_list ]
    return colors

def get_regimes(traj_df, outcome, RECOVER_MIN=12, RECOVER_MAX=25, seed=None):
    # SEGMENT/label trajectory timesteps: WARMUP, SEARCH, TRACK, RECOVER    
    # TRACK: Experienced odor within RECOVER_MAX steps
    # RECOVER: 

    if seed is not None and seed in config.seedmeta.keys():
        RECOVER_MIN = config.seedmeta[seed]['recover_min']
        RECOVER_MAX = config.seedmeta[seed]['recover_max']
        print("seed specific thresholds", RECOVER_MIN, RECOVER_MAX)

    # traj_df['regime'] = 'TRACK'
    traj_df['regime'] = 'RECOVER'

    # TRACK
    traj_df['regime'].loc[ traj_df['odor_lastenc'] <= RECOVER_MIN ] = 'TRACK'

    # SEARCH
    traj_df['regime'].loc[ traj_df['odor_lastenc'] >= RECOVER_MAX ] = 'SEARCH'

    # RECOVER: Between RECOVER_MIN and RECOVER_MAX
    recover_idxs = traj_df['odor_lastenc'].apply(lambda x: x in np.arange(RECOVER_MIN, RECOVER_MAX))
    traj_df['regime'].loc[ recover_idxs ] = 'RECOVER'

    # Warm-up in RECOVER
    traj_df['regime'].iloc[:RECOVER_MIN] = 'RECOVER' 

    # for i in range(traj_df.shape[0]):
    #     last_enc_i = traj_df['odor_lastenc'][i] 
    #     if last_enc_i <= RECOVER_MAX and last_enc_i >= RECOVER_MIN:
    #         idx_start = i - last_enc_i + RECOVER_MIN # 
    #         traj_df['regime'].iloc[ idx_start : i  ] = 'RECOVER'

    # idx = traj_df.shape[0]
    # while idx > 0:
    #     if traj_df['odor_lastenc'][idx] >= RECOVER_MAX:

    #         traj_df['regime'].iloc[ idx - traj_df['odor_lastenc'][idx] : idx] = 'RECOVER'

    # for i in range(traj_df.shape[0]):
    #     if traj_df['odor_lastenc'][i] >= RECOVER_MAX:
    #         traj_df['regime'].iloc[ i - traj_df['odor_lastenc'][i] : i] = 'RECOVER'


    # if traj_df['odor_01'][0] == 0:
    # traj_df['regime'].iloc[:RECOVER_MIN] = 'WARMUP'

    # RECOVER: First mark all off-plume times as RECOVER
    # traj_df['regime'].loc[ traj_df['odor_01'] == 0 ] = 'RECOVER'
    # for i in range(traj_df.shape[0]):
    #     if traj_df['odor_01'][i] == 0:
    #         traj_df['regime'].iloc[i] = 'RECOVER'
    #     else:
    #         break

    # RECOVER: Mark segments off plume that last longer than GAP_NEEDED timesteps
    # for i in range(i, traj_df.shape[0]):
    #     if traj_df['odor_lastenc'][i] >= RECOVER_MAX:
    #         traj_df['regime'].iloc[ i - traj_df['odor_lastenc'][i] : i] = 'RECOVER'

    # SEARCH: Mark the trailing part of trajectory for OOB
    # if outcome != 'HOME':
    #     for i in reversed(range(traj_df.shape[0])):
    #         if traj_df['odor_01'].iloc[i] == 0:
    #             traj_df['regime'].iloc[i] = 'SEARCH'
    #         else:
    #             traj_df['regime'].iloc[i:i+RECOVER_MAX] = 'RECOVER'
    #             break

    return traj_df['regime']

def get_traj_df(episode_log, 
    extended_metadata=False, 
    squash_action=False,
    n_history=20,
    seed=None,
    ):
    # squash_action=True only needed for old log files 
    # (this is now done in evalCli itself, during creation of episode_log)

    # Basic trajectory (x, y)
    trajectory = episode_log['trajectory']
    traj_df = pd.DataFrame( trajectory )  
    traj_df.columns = ['loc_x', 'loc_y']   

    # time
    traj_df['t_val'] = [record[0]['t_val'] for record in episode_log['infos']]

    # Observations & Actions
    obs = [ x[0] for x in episode_log['observations'] ]
    obs = pd.DataFrame(obs)
    obs = obs.iloc[:, -3:] # handles STACKING > 0
    obs.columns = ['wind_x', 'wind_y', 'odor']
    obs['wind_theta_obs'] = obs.apply(lambda row: wind_xy_to_theta(row['wind_x'], row['wind_y']), axis=1)
    traj_df['wind_theta_obs'] = shift_scale_theta(obs['wind_theta_obs'])
    traj_df['wind_x_obs'] = obs['wind_x']
    traj_df['wind_y_obs'] = obs['wind_y']


    act = episode_log['actions'] 
    act = pd.DataFrame(act)
    if squash_action:
        act = (np.tanh(act) + 1)/2
    act.columns = ['step', 'turn']
    traj_df['step'] = act['step']
    traj_df['turn'] = act['turn']

    traj_df['odor_obs'] = obs['odor']
    traj_df['odor_obs'] = [0. if x <= config.env['odor_threshold'] else x for x in traj_df['odor_obs']]

    traj_df['stray_distance'] = [record[0]['stray_distance'] for record in episode_log['infos']]

    # Observation derived
    traj_df['odor_01'] = [0 if x <= config.env['odor_threshold'] else 1 for x in traj_df['odor_obs']]
    traj_df['odor_clip'] = traj_df['odor_obs'].clip(lower=0., upper=1.0)

    # time since last encounter
    def _count_lenc(lenc0, maxcount=None):
        count = 0
        lenc = [0]
        for i in range(len(lenc0) - 1):
            count = 0 if lenc0[i] == 0 else count+1
            if maxcount is not None:
                count = maxcount if count >= maxcount else count
            lenc.append(count)
        return lenc
    # traj_df['odor_lastenc'] = _count_lenc( 1 - traj_df['odor_01'], maxcount=15 )
    traj_df['odor_lastenc'] = _count_lenc( 1 - traj_df['odor_01'], maxcount=None )

    ### REGIMEs
    outcome = episode_log['infos'][-1][0]['done'] 
    traj_df['regime'] = get_regimes(traj_df, outcome, seed=seed)
    ###

    # Other
    traj_df['agent_angle_ground'] = [ shift_scale_theta( 
        wind_xy_to_theta(record[0]['angle'][0], record[0]['angle'][1]) ) \
        for record in episode_log['infos']]
    traj_df['wind_angle_ground'] = [ shift_scale_theta( 
        wind_xy_to_theta(record[0]['wind_ground'][0], record[0]['wind_ground'][1]) ) for record in episode_log['infos']]
    traj_df['wind_speed_ground'] = [ np.linalg.norm(record[0]['wind_ground']) for record in episode_log['infos']]


    if extended_metadata:
        traj_df['t_val_norm'] = rescale_col(traj_df['t_val'], vmax=12) # 300/25 steps/fps

        # Observation derived
        # traj_df['odor_cummax'] = traj_df['odor_obs'].cummax()
        # traj_df['odor_cummax'] = rescale_col(traj_df['odor_cummax']) 

        # Add a range of ENC (encounters)
        for j in np.arange(2, n_history, step=2):
            j = int(j)
            colname = f'odor_enc_{j}'
            traj_df[colname] = traj_df['odor_01'].diff().fillna(0).clip(lower=0).ewm(span=j).mean()*25
            # traj_df[colname] = traj_df['odor_01'].diff().fillna(0).clip(lower=0).rolling(j).mean()*25
            # traj_df[colname] = traj_df['odor_01'].diff().fillna(0).clip(lower=0).ewm(span=j).mean()*25

        # Add a range of EWM
        traj_df['odor_ewm'] = traj_df['odor_clip'].ewm(span=15).mean()
        # traj_df['odor_ewm'] = rescale_col(traj_df['odor_ewm']) 
        for j in np.arange(2, n_history, step=2):
            j = int(j)
            colname = f'odor_ewm_{j}'
            traj_df[colname] = traj_df['odor_clip'].ewm(span=j).mean()
            # traj_df[colname] = traj_df['odor_clip'].ewm(span=j).mean()
            # traj_df[colname] = traj_df['odor_01'].ewm(span=j).mean()
            # traj_df[colname + '_norm'] = rescale_col(traj_df[colname]) 

        # Add a range of MA
        traj_df['odor_ma'] = traj_df['odor_clip'].rolling(15).mean()
        # traj_df['odor_ma_norm'] = rescale_col(traj_df['odor_ma']) 
        for j in np.arange(2, n_history, step=2):
            j = int(j)
            colname = f'odor_ma_{j}'
            traj_df[colname] = traj_df['odor_clip'].rolling(j).mean()
            # traj_df[colname] = traj_df['odor_01'].rolling(j).mean()
            # traj_df[colname + '_norm'] = rescale_col(traj_df[colname]) 

        traj_df['odor_lastenc_norm'] = rescale_col(traj_df['odor_lastenc']) 

        # Location derived
        traj_df['radius'] = [np.linalg.norm(record[0]['location']) for record in episode_log['infos']]
        # traj_df['radius_norm'] = rescale_col(traj_df['radius'], vmax=12) # Max dist + max stray

        # traj_df['stray_distance_norm'] = rescale_col(traj_df['stray_distance']) # Max stray distance

        traj_df['r_step'] = [record[0]['r_radial_step'] for record in episode_log['infos']]
        # traj_df['r_step_norm'] = rescale_col(traj_df['r_step']) 

        # Differences
        colnames_diff = [
            'loc_x', 
            'loc_y', 
            'wind_theta_obs', 
            'odor_obs', 
            'odor_01',
            'odor_clip', 
            'odor_lastenc', 
            'radius', 
            'stray_distance',
            'r_step', 
            'agent_angle_ground',
            'wind_angle_ground',
            'wind_speed_ground',
            ]
        for col in colnames_diff:
            traj_df[f'{col}_dt1'] = traj_df[col].diff()
            # traj_df[f'{col}_dt2'] = traj_df[f'{col}_dt1'].diff()

    return traj_df


# Trajectory DF --> Episode DF row
def get_episode_metadata(log, odor_threshold=ODOR_THRESHOLD, squash_action=False):    
    # Observation and Action information
    traj_df = get_traj_df(log, extended_metadata=False, squash_action=squash_action)
    ep_length = len(traj_df)
    
    turn_score = np.mean(np.abs(traj_df['turn'] - 0.5)) # More for more turns
    speed_score = np.mean(traj_df['step'])
    
    off_plume = traj_df['odor_obs'] < odor_threshold
    off_plume_fraction = 0 if ep_length==0 else np.sum(off_plume+0)/ep_length

    exit_rle = rle(off_plume+0)
    num_exits = np.sum(exit_rle[2] == 1)  # Num of exits from plume
    exit_durations = exit_rle[0][ exit_rle[2] == 1 ] # Lengths for when below threshold
    max_exit_duration = 0 if len(exit_durations)==0 else np.max(exit_durations)

    r_counts = traj_df['regime'].value_counts()
    for key in ['TRACK', 'RECOVER', 'SEARCH']:
        if key not in r_counts.keys():
            r_counts[key] = 0
    n_track = r_counts['TRACK']
    n_recover = r_counts['RECOVER']
    n_search = r_counts['SEARCH']

    # Other episode information
    infos = log['infos'][-1][0]
    
    return {
        'done': infos['done'],
        'ep_length': ep_length,
        'n_track': n_track,
        'n_recover': n_recover,
        'n_search': n_search,
        'off_plume_fraction': off_plume_fraction,
        'turn_score': turn_score,
        'speed_score': speed_score,
        'num_exits': num_exits,
        'max_exit_duration': max_exit_duration,

        'max_stray_distance': np.max(traj_df['stray_distance']),
        'max_odor_lastenc': np.max(traj_df['odor_lastenc']),
        'avg_stray_distance': np.mean(traj_df['stray_distance']),
        'avg_odor_lastenc': np.mean(traj_df['odor_lastenc']),
        'median_stray_distance': np.median(traj_df['stray_distance']),
        'median_odor_lastenc': np.median(traj_df['odor_lastenc']),

        'agent_angle_ground_median': np.median(traj_df['agent_angle_ground']),

        'radius_final': np.linalg.norm(infos['location']),
        'radius_covered': np.linalg.norm(infos['location_initial']) - np.linalg.norm(infos['location']),
        'end_x': infos['location'][0],
        'end_y': infos['location'][1],
        'start_x': infos['location_initial'][0],
        'start_y': infos['location_initial'][1],
        'reward': infos['reward'],
    }




def get_activity(log, is_recurrent, do_plot=False):
    if is_recurrent:
        ep_activity = pd.DataFrame(log['activity'])['rnn_hxs'].to_list()
    else:
        ep_activity = pd.DataFrame(log['activity'])['hx1_actor'].to_list()
    #     ep_activity = pd.DataFrame(log['activity'])['hidden_actor'].to_list()
    # ep_activity = pd.DataFrame(log['activity'])['h_actor'].to_list()
    ep_activity = np.stack(ep_activity)

    if do_plot:
        ep_activity.shape
        fig, ax = plt.subplots(figsize=(15,5))
        ms = ax.matshow(ep_activity.T, cmap='RdBu')
        # plt.gca().invert_xaxis()
        plt.ylabel(f"Neurons")
        plt.xlabel(f"Time-step")
        plt.title(f"Neural activity over Time/Trajectory")

        # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="1%", pad=0.05)
        plt.colorbar(ms, cax=cax)

    return ep_activity

def get_value(log):
    values = pd.DataFrame(log['activity'])['value'].to_list()
    values = np.array(values)
    return values


def pca_1episode(ep_activity, traj_df, twoD=True, threeD=False, colnames=None):
    if colnames is None:
        colnames = ['odor_obs', 
            'odor_01', 
            'odor_ma', 
            'odor_lastenc', 
            'wind_theta_obs', 
            'agent_angle_ground',
            'stray_distance', 
            'r_step', 
            ]

    pca = skld.PCA(3, whiten=False)
    pca.fit(ep_activity)
    X_pca = pca.transform(ep_activity)

    for colname in colnames:
      if twoD:
        fig = plt.figure(figsize=(5,5))
        ax = fig.gca()
        ax.plot(X_pca[:, 0], X_pca[:, 1],  linewidth=0.6, c='grey', alpha=0.5)
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                s=20, c=traj_df[colname], cmap=plt.cm.get_cmap('RdBu'), vmin=0., vmax=1.)
        plt.title(f"State Space [{colname}]")
        ax.scatter(X_pca[0, 0], X_pca[0, 1], c='g', marker='o', s=100) # Start
        ax.scatter(X_pca[-1, 0], X_pca[-1, 1], c='g', marker='x', s=200) # End
        ax.set_xlabel(f'PC1 (VarExp: {pca.explained_variance_ratio_[0]:0.2f})')
        ax.set_ylabel(f'PC2 (VarExp: {pca.explained_variance_ratio_[1]:0.2f})')
        ax.set_aspect('equal')
        
        # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(sc, cax=cax)

        plt.tight_layout()
        plt.show()

      if threeD:
        fig = plt.figure(figsize=(12,6))
        ax = fig.gca(projection='3d')
        ax.plot(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], linewidth=0.6, c='grey', alpha=0.5)
        sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                s=20, c=traj_df[colname], cmap=plt.cm.get_cmap('RdBu'), vmin=0, vmax=1)
        ax.scatter(X_pca[0, 0], X_pca[0, 1], X_pca[0, 2], c='g', marker='o', s=100) # Start
        ax.scatter(X_pca[-1, 0], X_pca[-1, 1], X_pca[-1, 2], c='g', marker='x', s=150) # End
        ax.set_xlabel(f'PC1 (VarExp: {pca.explained_variance_ratio_[0]:0.2f})')
        ax.set_ylabel(f'PC2 (VarExp: {pca.explained_variance_ratio_[1]:0.2f})')
        ax.set_zlabel(f'PC3 (VarExp: {pca.explained_variance_ratio_[2]:0.2f})')
        plt.title(f"State Space [{colname}]")
        
        # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        # plt.colorbar(sc) # Didn't work!
        plt.tight_layout()
        plt.show()

def tsne_1episode(ep_activity, twoD=True, threeD=False, colnames=None):
    # TODO
    from sklearn.manifold import TSNE
    tsne = TSNE(2)
    X_tsne = tsne.fit_transform(ep_activity)

    fig = plt.figure(figsize=(4,4))
    ax = fig.gca()
    ax.plot(X_tsne[:, 0], X_tsne[:, 1], linewidth=0.8)
    ax.scatter(X_tsne[0, 0], X_tsne[0, 1], c='r', marker='o', s=100) # Start
    ax.scatter(X_tsne[-1, 0], X_tsne[-1, 1], c='r', marker='x', s=150) # End
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    plt.show()


def animate_activity_1episode(ep_activity, traj_df, episode_idx, 
    outprefix, fprefix, pca_dims=2, pca_common=None):
    assert pca_dims in [2, 3]
    # TODO implement 3D

    if pca_common is None:
        pca = skld.PCA(pca_dims, whiten=False)
        pca.fit(ep_activity)
    else:
        print("Using pca_common...")
        pca = pca_common

    X_pca = pca.transform(ep_activity)

    t_vals =  traj_df['t_val'] 
    if not os.path.exists(f'{outprefix}/tmp/'):
        os.makedirs(f'{outprefix}/tmp/')

    output_fnames = []
    for t_idx in tqdm.tqdm(range(X_pca.shape[0])):
        t_val = t_vals[t_idx]
        title_text = f"ep:{episode_idx} step:{t_idx} [t:{t_val:0.2f}]"

        if pca_dims == 2:
            fig = plt.figure(figsize=(6,6))
            ax = fig.gca()
            ax.plot(X_pca[:, 0], X_pca[:, 1],  linewidth=0.6, c='grey', alpha=0.5)
            sc = ax.scatter(X_pca[:t_idx, 0], X_pca[:t_idx, 1], s=15, 
                c=np.arange(1, t_idx+1)/(t_idx+1), cmap=plt.cm.get_cmap('Reds'), vmin=0., vmax=1.)
            ax.scatter(X_pca[0, 0], X_pca[0, 1], c='g', marker='o', s=100) # Start
            ax.scatter(X_pca[t_idx, 0], X_pca[t_idx, 1], c='g', marker='x', s=200) # End
            ax.set_aspect('equal')
        else: 
            fig = plt.figure(figsize=(12,6))
            ax = fig.gca(projection='3d')
            ax.plot(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], linewidth=0.6, c='grey', alpha=0.5)
            sc = ax.scatter(X_pca[:t_idx, 0], X_pca[:t_idx, 1], X_pca[:t_idx, 2], s=15, 
                c=np.arange(1, t_idx+1)/(t_idx+1), cmap=plt.cm.get_cmap('Reds'), vmin=0., vmax=1.)
            ax.scatter(X_pca[0, 0], X_pca[0, 1], X_pca[0, 2], c='g', marker='o', s=100) # Start
            ax.scatter(X_pca[t_idx, 0], X_pca[t_idx, 1], X_pca[t_idx, 2], c='g', marker='x', s=150) # End
            ax.set_zlabel(f'PC3 (VarExp: {pca.explained_variance_ratio_[2]:0.2f})')

        ax.set_xlabel(f'PC1 (VarExp: {pca.explained_variance_ratio_[0]:0.2f})')
        ax.set_ylabel(f'PC2 (VarExp: {pca.explained_variance_ratio_[1]:0.2f})')
        plt.title(title_text)
        # plt.tight_layout()

        common_suffix = '_common' if pca_common is not None else '' 
        output_fname = f'{outprefix}/tmp/{fprefix}_pca{pca_dims}d{common_suffix}_ep{episode_idx}_step{t_idx:05d}.png'
        output_fnames.append(output_fname)
        plt.savefig(output_fname, bbox_inches='tight')

    output_fnames = natsorted(output_fnames,reverse=False)
    clips = [ImageClip(f).set_duration(0.08) for f in output_fnames] # 
    concat_clip = concatenate_videoclips(clips, method="compose")
    fanim = f"{outprefix}/{fprefix}_pca{pca_dims}d{common_suffix}_ep{episode_idx:03d}.mp4"
    concat_clip.write_videofile(fanim, fps=15, verbose=False, logger=None)
    print("Saved", fanim)

    for f in output_fnames:
        # https://stackoverflow.com/questions/10840533/most-pythonic-way-to-delete-a-file-which-may-not-exist
        with contextlib.suppress(FileNotFoundError):
            os.remove(f)


def get_activity_diff():
    # Neural activity Diff
    ep_activityT_reordered_diff = np.diff(ep_activityT_reordered)
    print(ep_activityT_reordered_diff.shape, ep_activityT_reordered.shape)
    ms = plt.matshow(ep_activityT_reordered_diff, cmap='RdBu')
    plt.ylabel(f"Neurons")
    plt.xlabel(f"Time-step")
    plt.title(f"Neural activity DIFF over Time/Trajectory")
    # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="1%", pad=0.05)
    plt.colorbar(ms, cax=cax)

    # TODO
    h_diff_norms = np.linalg.norm(ep_activityT_reordered_diff, axis=0)
    pd.Series(h_diff_norms).plot(figsize=(12,2), title=r"$\nabla h_t$");

def plot_activity_obs():
    # TODO
    # ms = plt.matshow(traj_df.T, cmap='RdBu')
    ms = plt.matshow(obs.T - obs.T.mean(axis=0), cmap='RdBu')
    plt.xlabel(f"Time-step")
    plt.title(f"trajectory DF")
    # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", size="1%", pad=0.05)
    plt.colorbar(ms, cax=cax)


#### Cluster and Disentangle the Dynamics ####
def get_actvity_reordering(ep_activity, do_plot=False):
    # fig = plt.figure(figsize=(1,1))
    clusts = sns.clustermap(ep_activity)
    reordering = clusts.dendrogram_col.reordered_ind
    ep_activityT_reordered = ep_activity.T[reordering,:]

    if do_plot:     
        ms = plt.matshow(ep_activityT_reordered, cmap='RdBu')
        plt.ylabel(f"Neurons")
        plt.xlabel(f"Time-step")
        plt.title(f"Neural activity over Time/Trajectory")
        # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", size="1%", pad=0.05)
        plt.colorbar(ms, cax=cax)

    return reordering, ep_activityT_reordered

def plot_disentabngled_systems():
    pass
    # # Split activity matrix into 2-3 sub-dynamical-systems
# split_idx = 14
# sys = []
# sys.append( ep_activityT_reordered[:split_idx,] )
# sys.append( ep_activityT_reordered[split_idx:,] )

# # sys = []
# # sys.append( ep_activityT_reordered[:7,] )
# # sys.append( ep_activityT_reordered[7:17] )
# # sys.append( ep_activityT_reordered[17:] )




# for si in range(len(sys)):
#     s = sys[si]
#     pca = skld.PCA(3, whiten=False)
#     pca.fit(s.T)
#     X_pca = pca.transform(s.T)

#     # PC1 x PC2
#     fig = plt.figure(figsize=(5,5))
#     ax = fig.gca()
#     ax.plot(X_pca[:, 0], X_pca[:, 1],  linewidth=0.6, c='grey', alpha=0.9)
#     ax.scatter(X_pca[0, 0], X_pca[0, 1], c='g', marker='o', s=100) # Start
#     ax.scatter(X_pca[-1, 0], X_pca[-1, 1], c='g', marker='x', s=200) # End
#     ax.set_xlabel(f'PC1 (VarExp: {pca.explained_variance_ratio_[0]:0.2f})')
#     ax.set_ylabel(f'PC2 (VarExp: {pca.explained_variance_ratio_[1]:0.2f})')
#     ax.set_aspect('equal')
#     colname = 'wind_theta' if si is 1 else 'odors_ma'
#     sc = ax.scatter(X_pca[:, 0], X_pca[:, 1],
#         s=20, c=traj_df[colname], cmap=plt.cm.get_cmap('RdBu'), vmin=0., vmax=1.)    
#     plt.title(f"RNN state - Sys{si} [{colname}]")
#     # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     plt.colorbar(sc, cax=cax)
#     plt.tight_layout()
#     plt.show()
    
    
#     # PC2 x PC3
#     fig = plt.figure(figsize=(5,5))
#     ax = fig.gca()
#     ax.plot(X_pca[:, 1], X_pca[:, 2],  linewidth=0.6, c='grey', alpha=0.9)
#     ax.scatter(X_pca[0, 1], X_pca[0, 2], c='g', marker='o', s=100) # Start
#     ax.scatter(X_pca[-1, 1], X_pca[-1, 2], c='g', marker='x', s=200) # End
#     ax.set_xlabel(f'PC2 (VarExp: {pca.explained_variance_ratio_[1]:0.2f})')
#     ax.set_ylabel(f'PC3 (VarExp: {pca.explained_variance_ratio_[2]:0.2f})')
#     ax.set_aspect('equal')
#     colname = 'wind_theta' if si is 1 else 'odors_ma'
#     sc = ax.scatter(X_pca[:, 1], X_pca[:, 2],
#         s=20, c=traj_df[colname], cmap=plt.cm.get_cmap('RdBu'), vmin=0., vmax=1.)    
#     plt.title(f"RNN state - Sys{si} [{colname}]")
#     # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     plt.colorbar(sc, cax=cax)
#     plt.tight_layout()
#     plt.show()
    

    
#     # 3D
#     fig = plt.figure(figsize=(12,6))
#     ax = fig.gca(projection='3d')
#     ax.plot(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], linewidth=0.6, c='grey', alpha=0.5)
#     sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
#             s=20, c=traj_df[colname], cmap=plt.cm.get_cmap('RdBu'), vmin=0, vmax=1)
    
#     ax.scatter(X_pca[0, 0], X_pca[0, 1], X_pca[0, 2], c='g', marker='o', s=100) # Start
#     ax.scatter(X_pca[-1, 0], X_pca[-1, 1], X_pca[-1, 2], c='g', marker='x', s=150) # End
#     ax.set_xlabel(f'PC1 (VarExp: {pca.explained_variance_ratio_[0]:0.2f})')
#     ax.set_ylabel(f'PC2 (VarExp: {pca.explained_variance_ratio_[1]:0.2f})')
#     ax.set_zlabel(f'PC3 (VarExp: {pca.explained_variance_ratio_[2]:0.2f})')
#     plt.title(f"RNN state - Sys{si} [{colname}]")
    
#     # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
# #     plt.colorbar(sc)

#     plt.tight_layout()
#     plt.show()

# From: https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi



    