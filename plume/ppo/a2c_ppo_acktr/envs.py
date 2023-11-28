import os
import numpy as np
import torch
import gym
from gym.spaces.box import Box
# import gymnasium as gym
# from gymnasium.spaces import Box

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnvWrapper, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_

import sys, os
sys.path.append('../')
sys.path.append('../../')

import importlib

def make_env(env_id, seed, rank, log_dir, allow_early_resets, args=None):
    # both seed info is redundant... seed args and provided separately
    if args.dynamic:
        import plume_env_dynamic as plume_env
        # from plume_env_dynamic import PlumeEnvironment, PlumeFrameStackEnvironment
        importlib.reload(plume_env)
        print("Using Dynamic Plume...")
    else:
        # hard coded to be false in evalCli
        import plume_env
        # from plume_env import PlumeEnvironment, PlumeFrameStackEnvironment
        importlib.reload(plume_env)
        print("Using Precomputed Plume...")

    def _thunk():
        if 'plume' in env_id:
            # hard coded to be plume in evalCli: env_name=plume. Only instance of env_id found so far
            if args.recurrent_policy or (args.stacking == 0):
                print("Using PlumeEnvironment...", flush=True, file=sys.stdout)
                env = plume_env.PlumeEnvironment(
                    dataset=args.dataset,
                    turnx=args.turnx,
                    movex=args.movex,
                    birthx=args.birthx,
                    birthx_max=args.birthx_max,
                    env_dt=args.env_dt,
                    loc_algo=args.loc_algo,
                    time_algo=args.time_algo,
                    diff_max=args.diff_max,
                    diff_min=args.diff_min,
                    auto_movex=args.auto_movex,
                    auto_reward=args.auto_reward,
                    walking=args.walking,
                    radiusx=args.radiusx,
                    r_shaping=args.r_shaping,
                    wind_rel=args.wind_rel,
                    action_feedback=args.action_feedback,
                    squash_action=args.squash_action,
                    flipping=args.flipping,
                    odor_scaling=args.odor_scaling,
                    qvar=args.qvar,
                    stray_max=args.stray_max,
                    obs_noise=args.obs_noise,
                    act_noise=args.act_noise,
                    seed=args.seed, 
                    )
            else:
                # Dont ever see this in logs so far
                print("Using PlumeFrameStackEnvironment...", flush=True, file=sys.stdout)
                env = plume_env.PlumeFrameStackEnvironment(
                    n_stack=args.stacking,
                    masking=args.masking,
                    stride=args.stride if args.stride >= 1 else 'log',
                    dataset=args.dataset,
                    turnx=args.turnx,
                    movex=args.movex,
                    birthx=args.birthx,
                    birthx_max=args.birthx_max,
                    env_dt=args.env_dt,
                    loc_algo=args.loc_algo,
                    time_algo=args.time_algo,
                    diff_max=args.diff_max,
                    diff_min=args.diff_min,
                    auto_movex=args.auto_movex,
                    auto_reward=args.auto_reward,
                    walking=args.walking,
                    radiusx=args.radiusx,
                    r_shaping=args.r_shaping,
                    wind_rel=args.wind_rel,
                    action_feedback=args.action_feedback,
                    squash_action=args.squash_action,
                    flipping=args.flipping,
                    odor_scaling=args.odor_scaling,
                    qvar=args.qvar,
                    stray_max=args.stray_max,
                    obs_noise=args.obs_noise,
                    act_noise=args.act_noise,
                    seed=args.seed,
                    )
        else:
            env = gym.make(env_id)
        env.seed(seed + rank)        

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets)

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])
        return env

    return _thunk


def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  allow_early_resets,
                  args=None,
                  num_frame_stack=None):
    envs = [
        make_env(env_name, seed, i, log_dir, allow_early_resets, args)
        for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, ret=False)
        else:
            envs = VecNormalize(envs, gamma=gamma) # type(envs.action_space) = <class 'gym.spaces.box.Box'>
        
    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# # Can be used to test recurrent policies for Reacher-v2
# class MaskGoal(gym.ObservationWrapper):
#     def observation(self, observation):
#         if self.env._elapsed_steps > 0:
#             observation[-2:] = 0
#         return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True
    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs
    def train(self):
        self.training = True
    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
