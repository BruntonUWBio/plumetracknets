import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import gym
import pandas as pd
import matplotlib.pyplot as plt
import time

import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

import sim_analysis
import config

from moviepy.editor import ImageClip, concatenate_videoclips
from natsort import natsorted
import contextlib
import os

from pylab import figure, text, scatter, show

import log_analysis

import tqdm

######################################################################################
### Helper functions ###
# Evaluate agent
def evaluate_agent(agent, 
    env, 
    n_steps_max=200, 
    n_episodes=1, 
    verbose=1):
    np.set_printoptions(precision=4)
    # Rollouts
    episode_logs = []
    for episode in tqdm.tqdm(range(n_episodes)): 
        trajectory = []
        observations = []
        actions = []
        rewards = []
        infos = []

        obs = env.reset()
        reward = 0
        done = False
        cumulative_reward = 0
        for step in range(n_steps_max):  
            # Select next action
            action = agent.act(obs, reward, done)
            # print("step, action:", step, action)
            # Step environment w/ action
            # print(action)
            obs, reward, done, info = env.step(action) # obs: [1, 3] 

            trajectory.append( info[0]['location'] )
            observations.append( obs )
            actions.append( action )
            rewards.append( reward )
            infos.append( info )

            cumulative_reward += reward
            if verbose > 1:
                print("{}: Action: {}, Odor:{:.6f}, Wind:({:.2f}, {:.2f}) Reward:{}, Loc:{}, Angle:{}".format(
                    step+1, 
                    action, 
                    obs[0][2], # odor
                    obs[0][0], # wind_x
                    obs[0][1], # wind_y
                    reward, 
                    [ '%.2f' % elem for elem in info[0]['location'] ],
                    '%.2f' % np.rad2deg( np.angle(info[0]['angle'][0] + 1j*info[0]['angle'][1]) )
                ))    
            if done:
                break
        if verbose > 0:
            print("Episode {} stopped at {} steps with cumulative reward {}".format(episode + 1, step + 1, cumulative_reward))

        episode_log = {
            'trajectory': trajectory,
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'infos': infos,
        }
        episode_logs.append(episode_log)

    return episode_logs

######################################################################################
## Behavior Analysis ##
def visualize_single_episode(data_puffs, data_wind, traj_df, 
    episode_idx, zoom=1, t_val=None, title_text=None, output_fname=None, 
    show=True, colorby=None, vmin=0, vmax=1, plotsize=None, xlims=None, ylims=None, legend=True):
    scatter_size = 15
    plotsize = (8,8) if plotsize is None else plotsize

    try:      
        fig, ax = sim_analysis.plot_puffs_and_wind_vectors(data_puffs, data_wind, t_val, 
                                           fname='', plotsize=plotsize, show=show)
    except Exception as e:
        print(episode_idx, e)
        return None, None

    # Crosshair at source
    plt.plot([0, 0],[-0.3,+0.3],'k-', linestyle = ":", lw=2)
    plt.plot([-0.3,+0.3],[0, 0],'k-', linestyle = ":", lw=2)

    # Handle custom colorby
    if colorby is not None and type(colorby) is not str:
        colors = colorby # assumes that colorby is a series
        colorby = 'custom'

    # Line for trajectory
    # linecolor = 'grey' if colorby is not None else 'red'
    # linecolor = 'black' if colorby is not None else 'red'
    linecolor='black'
    plt.plot(traj_df.iloc[:,0], traj_df.iloc[:,1], c=linecolor, lw=0.5) # Red line!
    ax.scatter(traj_df.iloc[0,0], traj_df.iloc[0,1], c='black', 
        edgecolor='black', marker='o', s=100) # Start

    # Scatter plot for odor/regime etc.
    # Default: colors indicate odor present/absent
    if colorby is None:
        colors = [config.traj_colormap['off'] if x <= config.env['odor_threshold'] else config.traj_colormap['on'] for x in traj_df['odor_obs']]
        cm = plt.cm.get_cmap('winter')
        plt.scatter(traj_df.iloc[:,0], traj_df.iloc[:,1], 
            c=colors, s=scatter_size, cmap=cm, vmin=vmin, vmax=vmax, alpha=1.0)
    if colorby is not None and colorby == 'complete': 
        # Colors indicate % trajectory complete
        colors = traj_df.index/len(traj_df)
        cm = plt.cm.get_cmap('winter')
        plt.scatter(traj_df.iloc[:,0], traj_df.iloc[:,1], 
            c=colors, s=scatter_size, cmap=cm, vmin=vmin, vmax=vmax, alpha=1.0)
    if colorby is not None and colorby == 'regime': 
        colors = [ config.regime_colormap[x] for x in traj_df['regime'].to_list() ]
        cm = None
        plt.scatter(traj_df.iloc[:,0], traj_df.iloc[:,1], 
            c=colors, s=scatter_size, cmap=cm, vmin=vmin, vmax=vmax, alpha=1.0)
    if colorby is not None and colorby == 'custom': 
        cm = plt.cm.get_cmap('winter')
        plt.scatter(traj_df.iloc[:,0], traj_df.iloc[:,1], 
            c=colors, s=scatter_size, cmap=cm, vmin=vmin, vmax=vmax, alpha=1.0)

    if zoom == 1: # Constant wind
        plt.xlim(-0.5, 10.5)
        plt.ylim(-1.5, +1.5)
    if zoom == 2: # Switch or Noisy
        plt.xlim(-1, 10.5)
        # plt.ylim(-5, 5)
        plt.ylim(-1.5, 5)
    if zoom == 3: # Walking
        plt.xlim(-0.15, 0.5)
        plt.ylim(-0.2, 0.2)
    if zoom == 4: # constant + larger arena
        plt.xlim(-0.5, 10.5)
        plt.ylim(-3, +3)
    if zoom == -1: # Adaptive -- fine for stills, jerky when used for animations
        plt.xlim(-0.5, 10.1)
        y_max = max(data_puffs[data_puffs.time == t_val].y.max(), traj_df.iloc[:,1].max()) + 0.5
        y_min = min(data_puffs[data_puffs.time == t_val].y.min(), traj_df.iloc[:,1].min()) - 0.5
        # print('y_max', data_puffs[data_puffs.time == t_val].y.max(), traj_df.loc[:,1].max())
        # print('y_min', data_puffs[data_puffs.time == t_val].y.min(), traj_df.loc[:,1].min())
        plt.ylim(y_min, y_max)
    if xlims is not None:
        plt.xlim(xlims[0], xlims[1])
    if ylims is not None:
        print(ylims)
        plt.ylim(ylims[0], ylims[1])

    # if title_text is not None and zoom > 0:
    #     plt.title(title_text)

    # plt.xticks([])
    # plt.yticks([])
    if zoom > 0:
        plt.xlabel('Arena length [m]')
        plt.ylabel('Arena width [m]')


    # Tweet
    # text(0.5, 0.92, title_text,
    #  horizontalalignment='center',
    #  verticalalignment='center',
    #  fontsize=12,
    #  transform = ax.transAxes)
    # plt.xticks([])
    # plt.yticks([])
    # plt.xlabel('')
    # plt.ylabel('')

    if legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        patch1 = mpatches.Patch(color=config.traj_colormap['off'], label='Off plume')   
        patch2 = mpatches.Patch(color=config.traj_colormap['on'], label='On plume')   
        handles.extend([patch1, patch2])
        # plt.legend(handles=handles, loc='upper left')
        # plt.legend(handles=handles, loc='lower right')
        leg = plt.legend(handles=handles, loc='upper right')
        # https://stackoverflow.com/questions/12848808/set-legend-symbol-opacity-with-matplotlib
        for lh in leg.legendHandles: 
            lh.set_alpha(1)

    if output_fname is not None:
        plt.savefig(output_fname, bbox_inches='tight')
    return fig, ax


def animate_single_episode(
    data_puffs, data_wind, traj_df, 
    t_vals, t_vals_all,
    episode_idx, outprefix, fprefix, zoom, 
    colorby=None, plotsize=None, legend=True):
    
    n_tvals = len(t_vals) 
    if n_tvals == 0:
       print("n_tvals == 0!") 
    output_fnames = [] 
    skipped_frames = 0
    if not os.path.exists(f'{outprefix}/tmp/'):
        os.makedirs(f'{outprefix}/tmp/')
        
    t_val_min = None
    for t_idx in tqdm.tqdm(range(n_tvals)):
        traj_df_subset = traj_df.iloc[:t_idx+1,:] # feed trajectory incrementally 
        t_val = t_vals[t_idx]
        if t_val_min is None:
            t_val_min = t_val
        if t_val not in t_vals_all: # TODO: HACK to skip when t_val missing in puff_data!!
            skipped_frames += 1
            continue
        output_fname = f'{outprefix}/tmp/{fprefix}_ep{episode_idx}_step{t_idx:05d}.png'
        output_fnames.append(output_fname)
        title_text = f"episode:{episode_idx} step:{t_idx+1} [t:{t_val:0.2f}]"
        # title_text = f"Step:{t_idx+1} [Time:{t_val:0.2f}]"
        # title_text = f"Time:{t_val:0.2f}s"
        title_text = f"Time:{t_val-t_val_min:0.2f}s"
        fig, ax = visualize_single_episode(data_puffs, data_wind, 
            traj_df_subset, 
            episode_idx, 
            zoom, 
            t_val=t_val, 
            title_text=title_text, 
            output_fname=output_fname,
            show=False,
            colorby=None,
            plotsize=plotsize,
            legend=legend,
            )
        
    if skipped_frames > 0:
        print(f"Skipped {skipped_frames} out of {n_tvals} frames!")
    output_fnames = natsorted(output_fnames,reverse=False)
    if len(output_fnames) == 0:
        print("No valid frames!")
        return

    clips = [ImageClip(f).set_duration(0.08) for f in output_fnames] # 
    concat_clip = concatenate_videoclips(clips, method="compose")
    fanim = f"{outprefix}/{fprefix}_ep{episode_idx:03d}.mp4"
    concat_clip.write_videofile(fanim, fps=15, verbose=False, logger=None)
    # fanim = f"{outprefix}/{fprefix}_ep{episode_idx:03d}.gif"
    # concat_clip.write_gif(fanim, fps=30, verbose=False, logger=None)
    print("Saved", fanim)
    
    for f in output_fnames:
        # https://stackoverflow.com/questions/10840533/most-pythonic-way-to-delete-a-file-which-may-not-exist
        with contextlib.suppress(FileNotFoundError):
            os.remove(f)


def visualize_episodes(episode_logs, 
                       zoom=1, 
                       outprefix=None, 
                       title_text=True, 
                       animate=False,
                       fprefix='trajectory',
                       dataset='constant',
                       birthx=1.0,
                       diffusionx=1.0,
                       episode_idxs=None,
                       colorby=None,
                       vmin=0, vmax=1,
                       plotsize=None,
                       legend=True,
                       ):

    # Trim/preprocess loaded dataset!
    t_starts = []
    t_ends = []
    for log in episode_logs: 
        t_starts.append( log['infos'][0][0]['t_val'] )
        t_ends.append( log['infos'][-1][0]['t_val'] )
    # try:
    #     radiusx = episode_logs[-1]['infos'][0][0]['radiusx']
    # except Exception as e:
    radiusx = 1.0    
    data_puffs_all, data_wind_all = sim_analysis.load_plume(dataset, 
        t_val_min=min(t_starts)-1.0, 
        t_val_max=max(t_ends)+1.0,
        radius_multiplier=radiusx,
        diffusion_multiplier=diffusionx,
        puff_sparsity=np.clip(birthx, a_min=0.01, a_max=1.00),
        )
    t_vals_all = data_puffs_all['time'].unique()

    # Plot and animate individual episodes
    n_episodes = len(episode_logs)
    if episode_idxs is None:
        episode_idxs = [i for i in range(n_episodes)]

    figs, axs = [], []
    for episode_idx in range(n_episodes): 
        episode_idx_title = episode_idxs[episode_idx] # Hack to take in custom idxs
        ep_log = episode_logs[episode_idx]
        trajectory = ep_log['trajectory']
        traj_df = pd.DataFrame( trajectory )
        traj_df.columns = ['loc_x', 'loc_y']   
        t_val_end = t_ends[episode_idx]
        traj_df['odor_obs'] = [o[0][-1] for o in ep_log['observations']] 
        # print(traj_df.shape)

        if title_text:
            title_text = f"ep:{episode_idx} t:{t_val_end:0.2f} "
            title_text += "step: {}".format(traj_df.shape[0])
        else:
            title_text = None

        if outprefix is not None:
            output_fname = f"{outprefix}/{fprefix}_{episode_idx_title:03}.png"
            print(output_fname)
        else:
            output_fname = None

        # Flip plume about x-axis (generalization)
        flipx = ep_log['infos'][0][0]['flipx']
        if flipx < 0:
            data_wind = data_wind_all.copy() # flip this per episode
            data_puffs = data_puffs_all.query("time <= @t_val_end + 1").copy()
            data_wind.loc[:,'wind_y'] *= flipx   
            data_puffs.loc[:,'y'] *= flipx 
        else:
            data_wind = data_wind_all.query("time <= @t_val_end + 1")
            data_puffs = data_puffs_all.query("time <= @t_val_end + 1")

        t_vals = [record[0]['t_val'] for record in ep_log['infos']]

        if colorby == 'regime': # HACK!
            assert False # not supported anymore; pass in list of colors
            # # print("visualize_episodes", colorby)
            # traj_df2 = log_analysis.get_traj_df(ep_log, 
            #                 extended_metadata=False, squash_action=False)
            # traj_df['regime'] = traj_df2['regime']
            # print("value_counts", traj_df['regime'].value_counts().to_dict())

        # Tweet
        # title_text = f"Time:{t_val:0.2f}s"
        # title_text = f"Time:{t_val-t_val_min:0.2f}s"


        ylims = xlims = None
        if zoom == 0:
            print("adaptive ylims")
            # xlims = [ traj_df['loc_x'].min() - 0.25, traj_df['loc_x'].max() + 0.25 ]
            xlims = [-0.5, 10.1]
            ylims = [ traj_df['loc_y'].min() - 0.25, traj_df['loc_y'].max() + 0.25 ]
        fig, ax = visualize_single_episode(data_puffs, data_wind, 
            traj_df, episode_idx_title, zoom, t_val_end, 
            title_text, output_fname, colorby=colorby,
            vmin=vmin, vmax=vmax, plotsize=plotsize, 
            xlims=xlims, ylims=ylims, legend=legend)
        figs.append(fig)
        axs.append(ax)

        if animate:
            animate_single_episode(data_puffs, data_wind, traj_df, 
                t_vals, t_vals_all, episode_idx_title, outprefix, 
                fprefix, zoom, colorby=colorby, plotsize=plotsize, legend=legend) 

    return figs, axs

def visualize_episodes_metadata(episode_logs, zoom=1, outprefix=None):
    n_episodes = len(episode_logs)
    for episode_idx in range(n_episodes): 
        # Plot Observations over time
        obs = [ x[0] for x in episode_logs[episode_idx]['observations'] ]
        obs = pd.DataFrame(obs)
        obs.columns = ['wind_x', 'wind_y', 'odor']
        obs['wind_theta'] = obs.apply(lambda row: wind_xy_to_theta(row['wind_x'], row['wind_y']), axis=1)
        # axs = obs.loc[:,['wind_theta','odor']].plot(subplots=True, figsize=(10,4), title='Observations over time')
        # axs[-1].set_xlabel("Timesteps")

        # Plot Actions over time
        act = [ x[0] for x in episode_logs[episode_idx]['actions'] ]
        act = pd.DataFrame(act)
        act.columns = ['step', 'turn']
        # axs = act.plot(subplots=True, figsize=(10,3), title='Actions over time')
        # axs[-1].set_xlabel("Timesteps")

        merged = pd.merge(obs, act, left_index=True, right_index=True)
        axs = merged.loc[:,['wind_theta','odor','step', 'turn']].plot(subplots=True, figsize=(10,5), title='Observations & Actions over time')
        axs[-1].set_xlabel("Timesteps")
        axs[0].set_ylim(-np.pi, np.pi)        
        # axs[1].set_ylim(0, 0.5)        
        # axs[2].set_ylim(0, 1)        
        # axs[3].set_ylim(-1, 1)        
        # plt.tight_layout()

        if outprefix is not None:
            fname = "{}_ep{}_meta.png".format(outprefix, episode_idx)
            plt.savefig(fname)
        plt.close()


#### Behavior Analysis ####
def sample_obs_action_pair(agent, off_plume=False):
    wind_relative_angle_radians = np.random.uniform(low=-np.pi, high=+np.pi)
    # Always in plume
    if off_plume:
        odor_observation = 0.0
    else:
        odor_observation = np.random.uniform(low=0.0, high=0.3) # Appropriate distribution?
    
    wind_observation = [ np.cos(wind_relative_angle_radians), np.sin(wind_relative_angle_radians) ]
    obs = np.array([wind_observation + [odor_observation] ]).astype(np.float32)
    action = agent.act(obs, reward=0, done=False)
    return np.concatenate([obs.flatten(), action.flatten()])
#     return [obs, action]

def get_samples(agent, N=1000, off_plume=False):
    samples = [ sample_obs_action_pair(agent, off_plume) for i in range(N) ]
    return samples


# Add a wind_theta column
def wind_xy_to_theta(x, y):
    return np.angle( x + 1j*y, deg=False )/np.pi # note div by np.pi!

def get_sample_df(agent, N=1000, off_plume=False):
    samples = get_samples(agent, N, off_plume)
    sample_df = pd.DataFrame(samples)
    sample_df.columns = ['wind_x', 'wind_y', 'odor', 'step', 'turn']
    sample_df['wind_theta'] = sample_df.apply(lambda row: wind_xy_to_theta(row['wind_x'], row['wind_y']), axis=1)
    return sample_df

def visualize_policy_from_samples(sample_df, outprefix=None):
    # Plot turning policy
    plt.figure(figsize=(1.1, 1.1))
    plt.scatter(sample_df['wind_theta'], sample_df['turn']-0.5, alpha=0.5, s=3)
    plt.xlabel("Wind angle [$\pi$]")
    plt.ylabel("Turn angle [$\pi$]")
    # plt.yticks([])
    # plt.title("Agent turn policy")
    plt.xlim(-1, +1)    
    plt.ylim(-1, +1)
    if outprefix is not None:
        fname = "{}_policy_turn.png".format(outprefix)
        plt.savefig(fname)
    # plt.close()

# agent_analysis.visualize_episodes_metadata([one_trial], zoom=1, outprefix=None)
# Copy-pasted from agent_analysis.py
def get_obs_act_for_episode(episode, plot=True, stacked=True):
    obs = [ x[0] for x in episode['observations'] ]
    obs = pd.DataFrame(obs)
    if stacked: # Stacked models
        obs = obs.iloc[:, -3:]
    obs.columns = ['wind_x', 'wind_y', 'odor']
    obs['wind_theta'] = obs.apply(lambda row: wind_xy_to_theta(row['wind_x'], row['wind_y']), axis=1)
    act = [ x[0] for x in episode['actions'] ]
    act = pd.DataFrame(act)
    act.columns = ['step', 'turn']
    merged = pd.merge(obs, act, left_index=True, right_index=True)
    return merged

######################################################################################
### Neural activity analysis ###
