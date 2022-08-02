"""
# Adapted from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

ENV=BipedalWalker-v3
NUMPROC=4 # 1800 FPS for 12, 3000 FPS for 24
python main.py --env-name $ENV --algo ppo --use-gae --log-interval 1 --num-steps 2048 --num-processes $NUMPROC --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits

ENV=BipedalWalkerHardcore-v3
NUMPROC=4
python main.py --env-name $ENV --algo ppo --recurrent-policy --num-mini-batch $NUMPROC --use-gae --log-interval 1 --num-steps 2048 --num-processes $NUMPROC --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits

###### DryRun Plume ###### 
SHAPE="step oob stray"
SHAPE=end
SHAPE=step
SHAPE="step stray"
SHAPE="xstep stray"
SHAPE=xstep
SHAPE="xstep ystray"
SHAPE="xstep stray_abs"
SHAPE="step oob"
SHAPE="step end"
ALGO=ppo
DATASET=constantx5b5
DATASET=noisy3x5b5
NUMPROC=4 # Walle... ~4h/model
NUMPROC=1 # Walle... ~4h/model

#  --eval_type short \
#  --eval_type skip \
#  --eval_type fixed \


RNNTYPE=VRNN
OUTSUFFIX=Test${RNNTYPE}$RANDOM
SAVEDIR=./trained_models/Test${RNNTYPE}/
mkdir -p $SAVEDIR
python -u main.py --env-name plume \
 --num-env-steps 10000 \
 --dataset $DATASET \
 --seed $RANDOM \
 --eval-interval 1 \
 --eval_type skip \
 --eval_episodes 3 \
 --hidden_size 64 \
 --r_shaping $SHAPE \
 --diff_max 0.9 \
 --diff_min 0.4 \
 --diffusion_max 1.0 \
 --diffusion_min 0.33 \
 --odor_scaling True \
 --flipping True \
 --qvar 0.0 \
 --birthx 0.2 \
 --weight_decay 0.01 \
 --obs_noise 0.025 \
 --act_noise 0.025 \
 --algo $ALGO \
 --recurrent-policy \
 --squash_action True \
 --test_episodes 10 \
 --viz_episodes 10 \
 --log-interval 1 \
 --save-interval 1 \
 --save-dir $SAVEDIR \
 --rnn_type ${RNNTYPE} \
 --num-processes $NUMPROC \
 --num-mini-batch $NUMPROC \
 --outsuffix $OUTSUFFIX \
 --use-gae --num-steps 2000 --lr 3e-4 --entropy-coef 0.005 --value-loss-coef 0.5 --ppo-epoch 10 --gamma 0.997 --gae-lambda 0.95 --use-linear-lr-decay
 --use-gae --num-steps 2000 --lr 3e-4 --entropy-coef 0.005 --value-loss-coef 0.5 --ppo-epoch 10 --gamma 0.99 --gae-lambda 0.95 --use-linear-lr-decay
 --use-gae --num-steps 500 --lr 3e-4 --entropy-coef 0.01 --value-loss-coef 0.5 --ppo-epoch 10 --gamma 0.99 --gae-lambda 0.95 --use-linear-lr-decay
 --use-gae --num-steps 100 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --gamma 0.99 --gae-lambda 0.95 --use-linear-lr-decay
 --use-gae --num-steps 2000 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --gamma 0.99 --gae-lambda 0.95 --use-linear-lr-decay

NUMPROC=2
OUTSUFFIX=TestMLPs10$RANDOM
SAVEDIR=./trained_models/${OUTSUFFIX}/
mkdir -p $SAVEDIR
python -u main.py --env-name plume \
 --stacking 10 \
  --save-dir $SAVEDIR \
 --squash_action True \
 --dataset $DATASET \
 --num-env-steps 10000 \
 --log-interval 1 \
 --save-interval 1 \
 --r_shaping $SHAPE \
 --algo $ALGO \
 --diff_max 0.5 \
 --num-processes $NUMPROC \
 --num-mini-batch $NUMPROC \
 --test_episodes 1 \
 --outsuffix $OUTSUFFIX \
 --odor_scaling True \
 --flipping True \
 --qvar 1.5 \
 --weight_decay 0.01 \
 --use-gae --num-steps 128 --lr 3e-4 --entropy-coef 0.01 --value-loss-coef 0.5 --ppo-epoch 10 --gamma 0.97 --gae-lambda 0.95 --use-linear-lr-decay

 --use-gae --num-steps 2048 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --gamma 0.99 --gae-lambda 0.95 --use-linear-lr-decay 

 --masking odor \
 --stride 5 \

"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
cuda_available = torch.cuda.is_available()

import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import pandas as pd

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.ppo import PPO
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate
import argparse
import json
import time

from setproctitle import setproctitle as ptitle
import evalCli 

def get_args():
    parser = argparse.ArgumentParser(description='PPO for Plume')
    parser.add_argument('--algo', default='ppo')
    parser.add_argument('--lr', type=float, default=7e-4,
        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False,
        help='use generalized advantage estimation')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=137,
        help='random seed (default: 1)')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--num-processes', type=int, default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=5,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--log-interval', type=int, default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
        help='disables CUDA training')
    parser.add_argument('--use-proper-time-limits', action='store_true', default=False,
        help='compute returns taking into account time limits')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
        help='use a recurrent policy')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False,
        help='use a linear schedule on the learning rate')

    # My params start
    parser.add_argument('--env-name')
    parser.add_argument('--log-dir', default='/tmp/gym/')
    parser.add_argument('--save-dir', default='./trained_models/')
    parser.add_argument('--dynamic', type=bool, default=False)
    parser.add_argument('--eval_type',  type=str, 
        default=['fixed', 'short', 'skip'][0])

    parser.add_argument('--eval_episodes', type=int, default=20)
    parser.add_argument('--eval-interval', type=int, default=None,
        help='eval interval, one eval per n updates (default: None)')


    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--rnn_type', type=str, default=None)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--betadist', type=bool, default=False)

    parser.add_argument('--stacking', type=int, default=0)
    parser.add_argument('--masking', type=str, default=None)
    parser.add_argument('--stride', type=int, default=1)

    # Curriculum hack
    parser.add_argument('--dataset', type=str, nargs='+', default=['constantx5b5'])
    parser.add_argument('--num-env-steps', type=int, nargs='+', default=[10e6])
    parser.add_argument('--qvar', type=float, nargs='+', default=[0.0])
    parser.add_argument('--birthx',  type=float, nargs='+', default=[1.0])
    parser.add_argument('--diff_max',  type=float, nargs='+', default=[0.8])
    parser.add_argument('--diff_min',  type=float, nargs='+', default=[0.4])

    parser.add_argument('--birthx_max',  type=float, default=1.0) # Only used for sparsity
    parser.add_argument('--dryrun',  type=bool, default=False)
    parser.add_argument('--curriculum', type=bool, default=False)
    parser.add_argument('--turnx',  type=float, default=1.0)
    parser.add_argument('--movex',  type=float, default=1.0)
    parser.add_argument('--auto_movex',  type=bool, default=False)
    parser.add_argument('--auto_reward',  type=bool, default=False)
    parser.add_argument('--loc_algo',  type=str, default='uniform')
    parser.add_argument('--time_algo',  type=str, default='uniform')
    parser.add_argument('--env_dt',  type=float, default=0.04)
    parser.add_argument('--outsuffix',  type=str, default='')
    parser.add_argument('--walking',  type=bool, default=False)
    parser.add_argument('--radiusx',  type=float, default=1.0)
    parser.add_argument('--diffusion_min',  type=float, default=1.0)
    parser.add_argument('--diffusion_max',  type=float, default=1.0)
    parser.add_argument('--r_shaping',  type=str, nargs='+', default=['step'])
    parser.add_argument('--wind_rel',  type=bool, default=True)
    parser.add_argument('--action_feedback',  type=bool, default=False)
    parser.add_argument('--squash_action',  type=bool, default=False)
    parser.add_argument('--flipping', type=bool, default=False)
    parser.add_argument('--odor_scaling', type=bool, default=False)
    parser.add_argument('--stray_max', type=float, default=2.0)
    parser.add_argument('--test_episodes',  type=int, default=50)
    parser.add_argument('--viz_episodes',  type=int, default=10)
    parser.add_argument('--model_fname',  type=str, default='')
    parser.add_argument('--obs_noise', type=float, default=0.0)
    parser.add_argument('--act_noise', type=float, default=0.0)

    args = parser.parse_args()

    # args.cuda = not args.no_cuda and 
    args.cuda = cuda_available
    print("CUDA:", args.cuda)
    assert args.algo in ['a2c', 'ppo']

    print(args)
    return args

def eval_lite(agent, env, args, device, actor_critic):
    # return None, None
    t_start = time.time()
    episode_summaries = []
    num_episodes = 0
    for i_episode in range(args.eval_episodes):
        recurrent_hidden_states = torch.zeros(1, 
                    actor_critic.recurrent_hidden_state_size, device=device)
        masks = torch.zeros(1, 1, device=device)
        obs = env.reset()

        reward_sum = 0
        ep_step = 0

        while True:
            with torch.no_grad():
                value, action, _, recurrent_hidden_states, activity = actor_critic.act(
                    obs, 
                    recurrent_hidden_states, 
                    masks, 
                    deterministic=True)

            obs, reward, done, info = env.step(action)
            masks.fill_(0.0 if done else 1.0)

            reward_sum += reward.detach().numpy().squeeze()
            ep_step += 1

            if done:
                num_episodes += 1
                episode_summary = {
                    'idx': i_episode,
                    'reward_sum': reward_sum,
                    'n_steps': ep_step,
                }
                episode_summaries.append(episode_summary)
                break # out of while loop

    episode_summaries = pd.DataFrame(episode_summaries)
    r_mean = episode_summaries['reward_sum'].mean()
    r_std = episode_summaries['reward_sum'].std()
    comp_time = time.time() - t_start        
    steps_mean = episode_summaries['n_steps'].mean()
    eval_record = {
        'r_mean': np.around(r_mean, decimals=2),
        'r_std': np.around(r_std, decimals=2),
        'steps_mean': np.around(steps_mean, decimals=2),
        't': np.around(comp_time, decimals=2),
    }
    return eval_record

def training_loop(agent, envs, args, device, actor_critic, 
    training_log=None, eval_log=None, eval_env=None):
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                          envs.observation_space.shape, envs.action_space,
                          actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=50) 
    best_mean = 0.0

    training_log = training_log if training_log is not None else []
    eval_log = eval_log if eval_log is not None else []

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):
        # print(f"On update {j} of {num_updates}")

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates, args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, activities = actor_critic.act(
                    rollouts.obs[step], 
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = args.save_dir # os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            args.model_fname = os.path.join(save_path, f'{args.env_name}_{args.outsuffix}.pt')
            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], args.model_fname)
            print('Saved', args.model_fname)

            current_mean = np.median(episode_rewards)
            if current_mean >= best_mean:
                best_mean = current_mean
                fname = os.path.join(save_path, f'{args.env_name}_{args.outsuffix}.pt.best')
                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
                ], fname)
                print('Saved', fname)

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Update {}/{}, T {}, FPS {}, {}-training-episode: mean/median {:.1f}/{:.1f}, min/max {:.1f}/{:.1f}"
                .format(j, num_updates, 
                        total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), 
                        np.min(episode_rewards),
                        np.max(episode_rewards))) 

            training_log.append({
                    'update': j,
                    'total_updates': num_updates,
                    'T': total_num_steps,
                    'FPS': int(total_num_steps / (end - start)),
                    'window': len(episode_rewards), 
                    'mean': np.mean(episode_rewards),
                    'median': np.median(episode_rewards), 
                    'min': np.min(episode_rewards),
                    'max': np.max(episode_rewards),
                })

            # Save training curve
            save_path = args.save_dir # os.path.join(args.save_dir, args.algo)
            os.makedirs(save_path, exist_ok=True)
            fname = os.path.join(save_path, f'{args.env_name}_{args.outsuffix}_{args.dataset}_train.csv')
            pd.DataFrame(training_log).to_csv(fname)


        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            if eval_env is not None:
                eval_record = eval_lite(agent, eval_env, args, device, actor_critic, )
                eval_record['T'] = total_num_steps
                eval_log.append(eval_record)
                print("eval_lite:", eval_record)

                save_path = args.save_dir # os.path.join(args.save_dir, args.algo)
                os.makedirs(save_path, exist_ok=True)
                fname = os.path.join(save_path, f'{args.env_name}_{args.outsuffix}_{args.dataset}_eval.csv')
                pd.DataFrame(eval_log).to_csv(fname)
        #     # ob_rms = utils.get_vec_normalize(envs).ob_rms
        #     ob_rms = None
        #     # evaluate(actor_critic, ob_rms, args.env_name, args.seed,
        #     #          args.num_processes, args.eval_log_dir, device)
        #     evaluate(actor_critic, args, device)
    return training_log, eval_log

def main():
    args = get_args()
    print("PPO Args --->", args)

    np.random.seed(args.seed)
    if args.betadist:
        print("Setting args.squash_action = False")
        args.squash_action = False # No squashing when using Beta

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    ptitle('PPO Seed {}'.format(args.seed))

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    args.eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(args.eval_log_dir)

    torch.set_num_threads(1)
    # gpu_idx = np.random.choice([i for i in range(torch.cuda.device_count())])
    gpu_idx = 0
    device = torch.device(f"cuda:{gpu_idx}" if args.cuda else "cpu")

    # Curriculum hack
    datasets = args.dataset
    birthxs = args.birthx
    qvars = args.qvar
    diff_maxs = args.diff_max
    diff_mins = args.diff_min
    num_env_stepss = args.num_env_steps
    assert len(datasets) == len(birthxs) 
    assert len(datasets) == len(qvars) 
    assert len(datasets) == len(diff_maxs) 
    assert len(datasets) == len(diff_mins) 
    assert len(datasets) == len(num_env_stepss) 
    stage_idx = 0
    training_log = None
    eval_log = None
    args.dataset = datasets[stage_idx] 
    args.birthx = birthxs[stage_idx] 
    args.qvar = qvars[stage_idx] 
    args.diff_max = diff_maxs[stage_idx] 
    args.diff_min = diff_mins[stage_idx] 
    args.num_env_steps = num_env_stepss[stage_idx] 


    envs = make_vec_envs(args.env_name, 
                        args.seed, 
                        args.num_processes,
                        args.gamma, 
                        args.log_dir, 
                        device, 
                        False, 
                        args)

    eval_env = make_vec_envs(
        args.env_name,
        args.seed + 1000,
        num_processes=1,
        gamma=args.gamma, 
        log_dir=args.log_dir, 
        device=device,
        args=args,
        allow_early_resets=True)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={
                     'recurrent': args.recurrent_policy,
                     'rnn_type': args.rnn_type,
                     'hidden_size': args.hidden_size,
                     },
        args=args)
    actor_critic.to(device)

    agent = PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay)

    # Save args and config info
    # https://stackoverflow.com/questions/16878315/what-is-the-right-way-to-treat-argparse-namespace-as-a-dictionary
    fname = f"{args.save_dir}/{args.env_name}_{args.outsuffix}_args.json"
    with open(fname, 'w') as fp:
        json.dump(vars(args), fp)


    # Save model at START of training
    fname = f'{args.save_dir}/{args.env_name}_{args.outsuffix}.pt.start'
    torch.save([
        actor_critic,
        getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
    ], fname)
    print('Saved', fname)


    # Curriculum hack
    num_stages = len(datasets)
    for stage_idx in range(num_stages):
        args.dataset = datasets[stage_idx] 
        args.birthx = birthxs[stage_idx] 
        args.qvar = qvars[stage_idx] 
        args.diff_max = diff_maxs[stage_idx] 
        args.diff_min = diff_mins[stage_idx] 
        args.num_env_steps = num_env_stepss[stage_idx] 
        print(f"Stage: {stage_idx}/{num_stages} - {args.dataset} b{args.birthx} q{args.qvar} n{args.num_env_steps}")

        if stage_idx > 0: # already made one above
            envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False, args)
        training_log, eval_log = training_loop(agent, envs, args, device, actor_critic, 
            training_log=training_log, eval_log=eval_log, eval_env=eval_env)        

    # Save model at END of training
    fname = f'{args.save_dir}/{args.env_name}_{args.outsuffix}.pt'
    torch.save([
        actor_critic,
        getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
    ], fname)
    print('Saved', fname)
    

    #### -------------- Done training - now Evaluate -------------- ####
    if args.eval_type == 'skip':
        return

    actor_critic.to('cpu')
    args.model_fname = fname

    # Evaluation
    print("Starting evaluation")
    datasets = ['switch15x5b5', 
                'switch30x5b5', 
                'switch45x5b5', 
                'constantx5b5', 
                'noisy3x5b5', 
                'noisy6x5b5']
    # if args.dataset not in datasets:
    #     datasets.append(args.dataset)
    #     datasets.reverse() # Do training data test first
    args.flipping = False
    args.dynamic = False
    args.fixed_eval = True if 'fixed' in args.eval_type else False
    args.birthx = 1.0
    args.birthx_max = 1.0
    args.qvar = 0.0 # doesn't matter for fixed
    args.obs_noise = 0.0
    args.act_noise = 0.0
    args.diffusion_max = args.diffusion_min # always test at min diffusion rate

    for ds in datasets:
      print(f"Evaluating on dataset: {ds}")
      args.dataset = ds
      test_sparsity = True if 'constantx5b5' in args.dataset else False
      test_sparsity = False if 'short' in args.eval_type else test_sparsity
      evalCli.eval_loop(args, actor_critic, test_sparsity=test_sparsity)

if __name__ == "__main__":
    main()
