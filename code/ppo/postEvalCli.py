"""
DIR=$(ls -d /home/satsingh/plume/plumezoo/latest/fly/memory/*VRNN*/ | head -n 1)
echo $DIR
python -u postEvalCli.py --model_dir $DIR --viz_episodes 2 


# Batch
MODELDIRS=$(ls -d /home/satsingh/plume/plumezoo/latest/fly/memory/*VRNN*seed3199993/)
MODELDIRS=$(ls -d /home/satsingh/plume/plumezoo/latest/fly/memory/*VRNN*seed9781ba/)
MODELDIRS=$(ls -d /home/satsingh/plume/plumezoo/latest/fly/memory/*VRNN*seed3307e9/)
MODELDIRS=$(ls -d /home/satsingh/plume/plumezoo/latest/fly/memory/*VRNN*/)
echo $MODELDIRS
for DIR in $MODELDIRS; do
    LOGFILE=${DIR}/posteval.log
    python -u postEvalCli.py --model_dir $DIR \
      --viz_episodes 20 >> $LOGFILE 2>&1 &
done
# --walking False # does NOT work that way!
#tail -f /home/satsingh/plume/plumezoo/latest/fly/memory/*VRNN*/posteval.log

# Sparse
for DIR in $MODELDIRS; do
    LOGFILE=${DIR}/posteval.log
    python -u postEvalCli.py --model_dir $DIR \
      --viz_episodes 20 --birthxs 0.4 >> $LOGFILE 2>&1 &
done

tail -f $LOGFILE

# Stitch videos side-by-side: see vid_stitch_cli for more options
for FNEURAL in $(find /home/satsingh/plume/plumezoo/latest/fly/memory/ -name "*pca3d_common_ep*.mp4"); do 
    FTRAJ=$(echo $FNEURAL | sed s/_pca3d_common//g) 
    # echo $FNEURAL $FTRAJ
    python -u ~/plume/plume2/vid_stitch_cli.py --fneural $FNEURAL --ftraj $FTRAJ
done

# Stitch videos side-by-side: see vid_stitch_cli for more options
MAXJOBS=20
for FNEURAL in $(find /home/satsingh/plume/plumezoo/latest/fly/memory/ -name "*pca3d_common_ep*.mp4"); do 
  while (( $(jobs -p | wc -l) >= MAXJOBS )); do sleep 10; done 
  FTRAJ=$(echo $FNEURAL | sed s/_pca3d_common//g) 
  # echo $FNEURAL $FTRAJ
  python -u ~/plume/plume2/vid_stitch_cli.py --fneural $FNEURAL --ftraj $FTRAJ &
done


"""
from __future__ import division
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=""

import torch
import argparse
import os
import sys
import numpy as np
import torch
import pandas as pd
import pickle

import glob
import pickle
from natsort import natsorted

import traceback

import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from pprint import pprint
import glob
import sys
sys.path.append('../')
# sys.path.append('/home/satsingh/plume/plume2/')
from plume_env import PlumeEnvironment, PlumeFrameStackEnvironment
import agents
import agent_analysis
import os
import log_analysis
import arch_utils as archu
import vid_stitch_cli

import sklearn.decomposition as skld


def post_eval(model_dir, use_datasets, n_episodes_home, n_episodes_other, viz_episodes):
    is_recurrent = True if ('GRU' in model_dir) or ('VRNN' in model_dir) else False

    selected_df = log_analysis.get_selected_df(model_dir, 
                                  use_datasets, 
                                  n_episodes_home=60, 
                                  n_episodes_other=60,
                                  min_ep_steps=0)


    # Generate common PCA
    h_episodes = []
    traj_dfs = []
    squash_action = True

    for episode_log in selected_df['log']:
        ep_activity = log_analysis.get_activity(episode_log, 
            is_recurrent, do_plot=False)
        h_episodes.append(ep_activity)
                
    h_episodes_stacked = np.vstack(h_episodes)
    # print(h_episodes_stacked.shape)

    pca_common = skld.PCA(3, whiten=False)
    pca_common.fit(h_episodes_stacked)

    # Get neural net 
    try:
        model_fname = model_dir[:-1] + ".pt"
        is_recurrent = True if ('GRU' in model_dir) or ('VRNN' in model_dir) else False
        actor_critic, ob_rms = \
            torch.load(model_fname, map_location=torch.device('cpu'))
        net = actor_critic.base.rnn #.weight_hh_l0.detach().numpy()
        J0 = net.weight_hh_l0.detach().numpy()
    except Exception as e:
        print(f"Exception: {e}")


    # Animate (1) trajectory, (2) Neural on common subspace, (3) eigen    
    # subset_df = selected_df.groupby(['dataset', 'outcome']).sample(viz_episodes)
    # subset_df = selected_df.query("outcome == 'HOME' and dataset == 'noisy3x5b5'").sample(viz_episodes)
    # subset_df = selected_df.query("outcome == 'HOME' and dataset == 'constantx5b5'").sample(viz_episodes)
    # subset_df = selected_df.query("outcome == 'HOME' and dataset == 'switch45x5b5'").sample(viz_episodes)
    # subset_df = selected_df.query("outcome == 'OOB' and dataset == 'noisy3x5b5'").sample(viz_episodes)
    # subset_df = selected_df.query("outcome == 'OOB' and dataset == 'constantx5b5'").sample(viz_episodes)
    # subset_df = selected_df.query("outcome == 'OOB' and dataset == 'switch45x5b5'").sample(viz_episodes)
    # subset_df = selected_df.query("dataset == 'noisy3x5b5'").groupby(['dataset', 'outcome']).sample(viz_episodes)
    # subset_df = selected_df.query("dataset == 'constantx5b5'").groupby(['dataset', 'outcome']).sample(viz_episodes)
    # subset_df = selected_df.query("dataset == 'switch45x5b5'").groupby(['dataset', 'outcome']).sample(viz_episodes)
    subset_df = selected_df.groupby(['dataset', 'outcome']).head(viz_episodes)
    for idx, row in subset_df.iterrows():
        if args.birthxs is not None: # HACK!!!!
            continue

        ep_activity = log_analysis.get_activity(row['log'], 
            is_recurrent, 
            do_plot=False)
        traj_df = log_analysis.get_traj_df(row['log'], 
                    extended_metadata=False, 
                    squash_action=squash_action)
        OUTPREFIX = model_dir
        dataset = row['dataset']
        outcome = row['outcome']
        fprefix = f'{row["dataset"]}_{outcome}'

        try:
            # Need to regenerate since no guarantee present already?
            zoom = 1 if 'constant' in dataset else 2    
            zoom = 4 if ('constant' in dataset) and ('HOME' not in outcome) else zoom 
            # zoom = 0 
            zoom = 3 if args.walking else zoom
            agent_analysis.visualize_episodes(episode_logs=[row['log']], 
                                              episode_idxs=[row['idx']],
                                              zoom=zoom, 
                                              dataset=row['dataset'],
                                              animate=True,
                                              fprefix=fprefix,
                                              diffusionx=args.diffusionx,
                                              outprefix=OUTPREFIX,
                                             )    

            log_analysis.animate_activity_1episode(ep_activity, 
                    traj_df, 
                    row['idx'], 
                    fprefix=fprefix,
                    outprefix=OUTPREFIX,
                    pca_dims=3,
                    pca_common=pca_common)

            # eig animations/plots
            eig_df = archu.get_eig_df_episode(net, row['log'])
            fname_suffix = f"{fprefix}_ep{row['idx']}"
            archu.animate_Jh_episode(eig_df, 
                fname_suffix=fname_suffix, 
                outprefix=OUTPREFIX)
            eig_vals, eig_vecs = np.linalg.eig(J0)
            archu.plot_eigvec_projections(eig_vals, 
                eig_vecs, 
                ep_activity, 
                fname_suffix=fname_suffix, 
                outprefix=OUTPREFIX)

        except Exception as e:
            print(f"Exception: {e}", traceback.print_exc())


    # DIRTY Hack to add sparse videos
    logfiles = natsorted(glob.glob(model_dir + '*.pkl'))
    if args.birthxs is not None:
        for birthx in args.birthxs:
            sparse_dataset = [f'constantx5b5_{birthx}']
            try:
                sparse_selected_df = log_analysis.get_selected_df(model_dir, 
                                  sparse_dataset, 
                                  n_episodes_home=60, 
                                  n_episodes_other=60,
                                  min_ep_steps=0)
            except Exception as e:
                print(f"Exception: {e}", traceback.print_exc())
                continue

            sparse_subset_df = sparse_selected_df.groupby(['dataset', 'outcome']).sample(viz_episodes)
            # sparse_subset_df = sparse_selected_df.query("outcome == 'OOB'").sample(viz_episodes)
            for idx, row in sparse_subset_df.iterrows():
                ep_activity = log_analysis.get_activity(row['log'], 
                    is_recurrent, 
                    do_plot=False)
                traj_df = log_analysis.get_traj_df(row['log'], 
                            extended_metadata=False, 
                            squash_action=squash_action)
                OUTPREFIX = model_dir
                dataset = row['dataset'].split('_')[0]
                outcome = row['outcome']
                fprefix = f'{row["dataset"]}_{outcome}'
                print("dataset",dataset)

                try:
                    # Need to regenerate since no guarantee present already?
                    zoom = 1 if 'constant' in dataset else 2    
                    zoom = 4 if ('constant' in dataset) and ('HOME' not in outcome) else zoom 
                    # zoom = 0 
                    zoom = 3 if args.walking else zoom
                    agent_analysis.visualize_episodes(episode_logs=[row['log']], 
                                                      episode_idxs=[row['idx']],
                                                      zoom=zoom, 
                                                      dataset=dataset,
                                                      animate=True,
                                                      fprefix=fprefix,
                                                      outprefix=OUTPREFIX,
                                                      birthx=float(birthx),
                                                      diffusionx=args.diffusionx,
                                                     )    

                    log_analysis.animate_activity_1episode(ep_activity, 
                            traj_df, 
                            row['idx'], 
                            fprefix=fprefix,
                            outprefix=OUTPREFIX,
                            pca_dims=3,
                            pca_common=pca_common)

                    # eig animations/plots
                    eig_df = archu.get_eig_df_episode(net, row['log'])
                    fname_suffix = f"{fprefix}_ep{row['idx']}"
                    archu.animate_Jh_episode(eig_df, 
                        fname_suffix=fname_suffix, 
                        outprefix=OUTPREFIX)
                    eig_vals, eig_vecs = np.linalg.eig(J0)
                    archu.plot_eigvec_projections(eig_vals, 
                        eig_vecs, 
                        ep_activity, 
                        fname_suffix=fname_suffix, 
                        outprefix=OUTPREFIX)

                except Exception as e:
                    print(f"Exception: {e}", traceback.print_exc())


### MAIN ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Common neural subspace plots/animations')
    parser.add_argument('--model_dir', default=None)
    parser.add_argument('--viz_episodes', type=int, default=2)
    parser.add_argument('--walking', type=bool, default=False)
    parser.add_argument('--birthxs', type=str, nargs='+', default=None)
    parser.add_argument('--diffusionx',  type=float, default=1.0)

    args = parser.parse_args()
    print(args)
    use_datasets = ['constantx5b5', 'switch45x5b5', 'noisy3x5b5']
    n_episodes_home = 30
    n_episodes_other = 30

    post_eval(args.model_dir, 
        use_datasets, 
        n_episodes_home, 
        n_episodes_other, 
        args.viz_episodes)