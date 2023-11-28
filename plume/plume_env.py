import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import pandas as pd
import numpy as np
import gym
from gym import spaces
# import gymnasium as gym
# from gymnasium.spaces import Box

import sim_analysis
# import tqdm # not used
from pprint import pprint

import config
from scipy.spatial.distance import cdist 

class PlumeEnvironment(gym.Env):
  """
  Documentation: https://gym.openai.com/docs/#environments
  Plume tracking
  """
  def __init__(self, 
    t_val_min=60.00, 
    sim_steps_max=300, # steps
    reset_offset_tmax=30, # seconds; max secs for initial offset from t_val_min
    dataset='constantx5b5',
    move_capacity=2.0, # Max agent speed in m/s
    turn_capacity=6.25*np.pi, # Max agent CW/CCW turn per second
    wind_obsx=1.0, # normalize/divide wind observations by this quantity (move_capacity + wind_max) 
    movex=1.0, # move_max multiplier for tuning
    turnx=1.0, # turn_max multiplier for tuning
    birthx=1.0, # per-episode puff birth rate sparsity minimum
    birthx_max=1.0, # overall odor puff birth rate sparsity max
    env_dt=0.04,
    loc_algo='quantile',
    qvar=1.0, # Variance of init. location; higher = more off-plume initializations
    time_algo='uniform',
    angle_algo='uniform',
    homed_radius=0.2, # meters, at which to end flying episode
    stray_max=2.0, # meters, max distance agent can stray from plume
    wind_rel=True, # Agent senses relative wind speed (not ground speed)
    auto_movex=False, # simple autocurricula for movex
    auto_reward=False, # simple autocurricula for reward decay
    diff_max=0.8, # teacher curriculum; sets the quantile of init x location 
    diff_min=0.4, # teacher curriculum; sets the quantile of init x location 
    r_shaping=['step', 'oob'], # 'step', 'end'
    rewardx=1.0, # scale reward for e.g. A3C
    rescale=False, # rescale/normalize input/outputs [redundant?]
    squash_action=False, # apply tanh and rescale (useful with PPO)
    walking=False,
    walk_move=0.05, # m/s (x100 for cm/s)
    walk_turn=1.0*np.pi, # radians/sec
    radiusx=1.0, 
    diffusion_min=1.00, 
    diffusion_max=1.00, 
    action_feedback=False,
    flipping=False, # Generalization/reduce training data bias
    odor_scaling=False, # Generalization/reduce training data bias
    obs_noise=0.0, # Multiplicative: Wind & Odor observation noise.
    act_noise=0.0, # Multiplicative: Move & Turn action noise.
    dynamic=False,
    seed=137,
    verbose=0):
    super(PlumeEnvironment, self).__init__()

    assert dynamic is False
    np.random.seed(seed)    
    self.arguments = locals()
    print("PlumeEnvironment:", self.arguments)
    
    self.verbose = verbose
    self.venv = self
    self.walking = walking
    self.rewardx = rewardx
    self.rescale = rescale
    self.odor_scaling = odor_scaling
    self.stray_max = stray_max
    self.wind_obsx = wind_obsx
    self.reset_offset_tmax = reset_offset_tmax
    self.action_feedback = action_feedback
    self.qvar = qvar
    self.squash_action = squash_action
    self.obs_noise = obs_noise
    self.act_noise = act_noise
    if self.squash_action:
        print("Squashing actions to 0-1")

    # Fixed evaluation related:
    self.fixed_time_offset = 0.0 # seconds
    self.fixed_angle = 0.0 # downwind
    self.fixed_x = 7.0 
    self.fixed_y = 0.0 # might not work for switch/noisy! 


    # Environment/state variables
    # self.dt = config.env['dt'] 
    self.dt = env_dt # 0.1, 0.2, 0.4, 0.5 sec
    # self.fps = config.env['fps'] # 20/25/50/100 steps/sec
    self.fps = int(1/self.dt)
    self.sim_fps = 100
    self.episode_step = 0 # skip_steps done during loading

    # Load simulated data
    self.radiusx = radiusx
    self.birthx = birthx
    self.birthx_max = birthx_max
    self.diffusion_max = diffusion_max # Puff diffusion multiplier (initial)
    self.diffusion_min = diffusion_min # Puff diffusion multiplier (reset-time)
    self.t_val_min = t_val_min
    self.episode_steps_max = sim_steps_max # Short training episodes to gather rewards
    self.t_val_max = self.t_val_min + self.reset_offset_tmax + 1.0*self.episode_steps_max/self.fps + 1.00

    self.set_dataset(dataset)

    # Correction for short simulations
    if self.data_wind.shape[0] < self.episode_steps_max:
      if self.verbose > 0:
        print("Wind data available only up to {} steps".format(self.data_wind.shape[0]))
      self.episode_steps_max = self.data_wind.shape[0]

    # Other initializations -- many redundant, see .reset() 
    # self.agent_location = np.array([1, 0]) # TODO: Smarter
    self.agent_location = None
    self.agent_location_last = self.agent_location
    self.agent_location_init = self.agent_location
    random_angle = np.pi * np.random.uniform(0, 2)
    self.agent_angle_radians = [np.cos(random_angle), np.sin(random_angle)] # Sin and Cos of angle of orientation
    self.step_offset = 0 # random offset per trial in reset()
    self.t_val = self.t_vals[self.episode_step + self.step_offset] 
    self.tidx = self.tidxs[self.episode_step + self.step_offset] 
    self.tidx_min_episode = self.tidx
    self.tidx_max_episode = self.tidx
    self.wind_ground = None
    self.stray_distance = 0
    self.stray_distance_last = 0
    self.agent_velocity_last = np.array([0, 0]) # Maintain last timestep velocity (in absolute coordinates) for relative sensory observations
    self.episode_reward = 0

    # Generalization & curricula
    self.r_shaping = r_shaping
    print("Reward Shaping", self.r_shaping)
    self.flipping = flipping 
    self.flipx = 1.0 # flip puffs around x-axis? 
    self.difficulty = diff_max # Curriculum
    self.diff_max = diff_max # Curriculum
    self.diff_min = diff_min # Curriculum
    self.odorx = 1.0 # Doesn't make a difference except when thresholding
    self.turnx = turnx
    self.movex = movex
    self.auto_movex = auto_movex
    self.auto_reward = auto_reward
    self.reward_decay = 1.00
    self.loc_algo = loc_algo
    self.angle_algo = angle_algo
    self.time_algo = time_algo
    assert self.time_algo in ['uniform', 'linear', 'fixed']
    self.outcomes = [] # store outcome last N episodes

    # Constants
    self.wind_rel = wind_rel
    self.turn_capacity = turn_capacity
    self.move_capacity = move_capacity 
    # self.turn_capacity = 1.0 * np.pi # Max agent can turn CW/CCW in one timestep
    # self.move_capacity = 0.025 # Max agent can move in one timestep
    self.arena_bounds = config.env['arena_bounds'] 
    self.homed_radius = homed_radius  # End session if dist(agent - source) < homed_radius
    self.rewards = {
      'tick': -10/self.episode_steps_max,
      'homed': 101.0,
      }


    # Define action and observation spaces
    # Actions:
    # Move [0, 1], with 0.0 = no movement
    # Turn [0, 1], with 0.5 = no turn... maybe change to [-1, 1]
    self.action_space = spaces.Box(low=0, high=+1,
                                        shape=(2,), dtype=np.float32)
    if self.rescale:
        ## Rescaled to [-1,+1] to follow best-practices: 
        # https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html#tips-and-tricks-when-creating-a-custom-environment
        # Both will first clip to [-1,+1] then map to [0,1] with all other code remaining same
        self.action_space = spaces.Box(low=-1, high=+1,
                                        shape=(2,), dtype=np.float32)

    # Observations
    # Wind velocity [-1, 1] * 2, Odor concentration [0, 1]
    obs_dim = 3 if not self.action_feedback else 3+2
    self.observation_space = spaces.Box(low=-1, high=+1,
                                        shape=(obs_dim,), dtype=np.float32)

    ######## Experimental "walking mode" ########
    if self.walking:
        self.turn_capacity = walk_turn 
        self.move_capacity = walk_move 
        self.homed_radius = 0.02 # m i.e. 18cm walk from 0.20m (flying "homed" distance)
        self.stray_max = 0.05 # meters
        # self.rewards['tick'] = -1/self.episode_steps_max

  def update_env_param(self, params):
      for k,v in params.items():
          setattr(self, k, v)

  def set_dataset(self, dataset):
    self.dataset = dataset
    self.data_puffs_all, self.data_wind_all = sim_analysis.load_plume(
        dataset=self.dataset, 
        t_val_min=self.t_val_min, 
        t_val_max=self.t_val_max,
        env_dt=self.dt,
        puff_sparsity=np.clip(self.birthx_max, a_min=0.01, a_max=1.00),
        diffusion_multiplier=self.diffusion_max,
        radius_multiplier=self.radiusx,
        )
    if self.walking:
        self.data_puffs_all = self.data_puffs_all.query('x <= 0.5')
    self.data_puffs = self.data_puffs_all.copy() # trim this per episode
    self.data_wind = self.data_wind_all.copy() # trim/flip this per episode
    self.t_vals = self.data_wind['time'].tolist()
    print("wind: t_val_diff", (self.t_vals[2] - self.t_vals[1]), "env_dt", self.dt)
    t_vals_puffs = self.data_puffs['time'].unique()
    print("puffs: t_val_diff", (t_vals_puffs[2] - t_vals_puffs[1]), "env_dt", self.dt)
    self.tidxs = self.data_wind['tidx'].tolist()

  def reload_dataset(self):
    self.set_dataset(self.dataset)

  def set_difficulty(self, level, verbose=True): # Curriculum
    """
    Location distance as a form of curriculum learning
    :level: in [0.0, 1.0] with 0.0 being easiest
    """
    if level < 0:
        self.difficulty = self.diff_max
    else:
        level = np.clip(level, 0.0, 1.0)
        self.difficulty = level
    if verbose:
        print("set_difficulty to", self.difficulty)

  def sense_environment(self):
    if (self.verbose > 1) and (self.episode_step >= self.episode_steps_max): # Debug mode
        pprint(vars(self))

    # Wind
    wind_absolute = self.wind_ground # updated by step()
    
    # Subtract agent velocity to convert to (observed) relative velocity
    if self.wind_rel: 
        wind_absolute = self.wind_ground - self.agent_velocity_last # TODO: variable should be named wind_relative

    # Get wind relative angle
    agent_angle_radians = np.angle( self.agent_angle[0] + 1j*self.agent_angle[1], deg=False )
    wind_angle_radians = np.angle( wind_absolute[0] + 1j*wind_absolute[1], deg=False )
    wind_relative_angle_radians = wind_angle_radians - agent_angle_radians
    wind_observation = [ np.cos(wind_relative_angle_radians), np.sin(wind_relative_angle_radians) ]    
    # Un-normalize wind observation by multiplying by magnitude
    wind_magnitude = np.linalg.norm(np.array( wind_absolute ))/self.wind_obsx
    wind_observation = [ x*wind_magnitude for x in wind_observation ] # convert back to velocity
    # Add observation noise
    wind_observation = [ x*(1.0+np.random.uniform(-self.obs_noise, +self.obs_noise)) for x in wind_observation ]

    if self.verbose > 1:
        print('wind_observation', wind_observation)
        print('t_val', self.t_val)

    # Odor
    # odor_observation = sim_analysis.get_concentration_at_point_in_time_pandas(
    #     self.data_puffs, self.t_val, self.agent_location[0], self.agent_location[1])
    odor_observation = sim_analysis.get_concentration_at_tidx(
        self.data_puffs, self.tidx, self.agent_location[0], self.agent_location[1])
    if self.verbose > 1:
        print('odor_observation', odor_observation)
    if self.odor_scaling:
        odor_observation *= self.odorx # Random scaling to improve generalization 
    odor_observation *= 1.0 + np.random.uniform(-self.obs_noise, +self.obs_noise) # Add observation noise

    odor_observation = 0.0 if odor_observation < config.env['odor_threshold'] else odor_observation
    odor_observation = np.clip(odor_observation, 0.0, 1.0) # clip

    # Return
    observation = np.array(wind_observation + [odor_observation]).astype(np.float32) # per Gym spec
    if self.verbose > 1:
        print('observation', observation)
    return observation

  def get_abunchofpuffs(self, max_samples=300):  
    # Z = self.data_puffs[self.data_puffs.time==self.t_val].loc[:,['x','y']]
    # Z = self.data_puffs[self.data_puffs.tidx==self.tidx].loc[:,['x','y']]
    Z = self.data_puffs.query(f"tidx == {self.tidx}").loc[:,['x','y']]
    Z = Z.sample(n=max_samples, replace=False) if Z.shape[0] > max_samples else Z
    return Z

  def get_stray_distance(self):
    Z = self.get_abunchofpuffs()
    Y = cdist(Z.to_numpy(), np.expand_dims(self.agent_location,axis=0), metric='euclidean')
    try:
        minY = min(Y) 
    except Exception as ex:
        print(f"Exception: {ex}, t:{self.t_val:.2f}, tidx:{self.tidx}({self.tidx_min_episode}...{self.tidx_max_episode}), ep_step:{self.episode_step}, {Z}")  
        minY = np.array([0])      
    return minY[0] # return float not float-array

  def get_initial_location(self, algo):
    loc_xy = None
    if 'uniform' in algo:
        loc_xy = np.array([
            2 + np.random.uniform(-1, 1), 
            np.random.uniform(-0.5, 0.5)])

        if self.walking:
            loc_xy = np.array([
              0.2 + np.random.uniform(-0.1, 0.1), 
              np.random.uniform(-0.05, 0.05)])

    if 'linear' in algo:
        # TODO
        loc_xy = np.array([
            2 + np.random.uniform(-1, 1), 
            np.random.uniform(-0.5, 0.5)])

    if 'quantile' in algo:
        """ 
        Distance curriculum
        Start the agent at a location with random location with mean and var
        decided by distribution/percentile of puffs 
        """
        q_curriculum = np.random.uniform(self.diff_min, self.diff_max)

        Z = self.get_abunchofpuffs()
        X_pcts = Z['x'].quantile([q_curriculum-0.1, q_curriculum]).to_numpy()
        X_mean, X_var = X_pcts[1], X_pcts[1] - X_pcts[0]
        # print("initial X mean, var, q: ", X_mean, X_var, q_curriculum)
        Y_pcts = Z.query("(x >= (@X_mean - @X_var)) and (x <= (@X_mean + @X_var))")['y'].quantile([0.05,0.5]).to_numpy()
        Y_pcts
        Y_mean, Y_var = Y_pcts[1], min(1, Y_pcts[1] - Y_pcts[0]) # TODO: What was min for?
        # print(Y_mean, Y_var)
        varx = self.qvar 
        # if 'switch' in self.dataset: # Preferably start within/close to plume
        #     varx = 0.1
        loc_xy = np.array([X_mean + varx*X_var*np.random.randn(), 
            Y_mean + varx*Y_var*np.random.randn()]) 

    if 'fixed' in algo:
        loc_xy = np.array( [self.fixed_x, self.fixed_y] )

    return loc_xy

  def get_initial_step_offset(self, algo):
    """ Time curriculum """
    if 'uniform' in algo:
        step_offset = int(self.fps * np.random.uniform(low=0.00, high=self.reset_offset_tmax))

    if 'linear' in algo:
        window = 5 # seconds
        mean = window + self.difficulty*(self.reset_offset_tmax-window)
        step_offset = int(self.fps * np.random.uniform(low=mean-window, high=mean+window))
        # print("mean, offset_linear:", mean, offset)

    if 'fixed' in algo: # e.g. fixed eval schedule
        step_offset = int(self.fps * self.fixed_time_offset)

    return step_offset

  def get_initial_angle(self, algo):
    if 'uniform' in algo:
        # Initialize agent to random orientation [0, 2*pi]
        random_angle = np.random.uniform(0, 2*np.pi)
        agent_angle = np.array([np.cos(random_angle), np.sin(random_angle)]) # Sin and Cos of angle of orientation
    if 'fixed' in algo: # e.g. fixed eval schedule
        agent_angle = np.array([np.cos(self.fixed_angle), np.sin(self.fixed_angle)]) # Sin and Cos of angle of orientation
    return agent_angle

  def diffusion_adjust(self, diffx):
    min_radius = 0.01
    self.data_puffs.loc[:,'radius'] -= min_radius # subtract initial radius
    self.data_puffs.loc[:,'radius'] *= diffx/self.diffusion_max  # adjust 
    self.data_puffs.loc[:,'radius'] += min_radius # add back initial radius
    # Fix other columns
    self.data_puffs['x_minus_radius'] = self.data_puffs.x - self.data_puffs.radius
    self.data_puffs['x_plus_radius'] = self.data_puffs.x + self.data_puffs.radius
    self.data_puffs['y_minus_radius'] = self.data_puffs.y - self.data_puffs.radius
    self.data_puffs['y_plus_radius'] = self.data_puffs.y + self.data_puffs.radius
    self.data_puffs['concentration'] = (min_radius/self.data_puffs.radius)**3

  def reset(self):
    """
    return Gym.Observation
    """
    # print(f'reset() called; self.birthx = {self.birthx}', flush=True)
    self.episode_reward = 0
    self.episode_step = 0 # skip_steps already done during loading
    # Add randomness to start time PER TRIAL!
    self.step_offset = self.get_initial_step_offset(self.time_algo)
    self.t_val = self.t_vals[self.episode_step + self.step_offset] 
    self.t_val_max_episode = self.t_val + 1.0*self.episode_steps_max/self.fps + 1.0
    self.tidx = self.tidxs[self.episode_step + self.step_offset] # Use tidx when possible
    self.tidx_min_episode = self.tidx
    self.tidx_max_episode = self.tidx + self.episode_steps_max*int(100/self.fps) + self.fps 

    # SPEEDUP (subset puffs to those only needed for episode)
    # self.data_puffs = self.data_puffs_all.query('(time > @self.t_val-1) and (time < @self.t_val_max_episode)') # Speeds up queries!
    self.data_puffs = self.data_puffs_all.query('(tidx >= @self.tidx-1) and (tidx <= @self.tidx_max_episode)') # Speeds up queries!

    # Dynamic birthx for each episode
    if self.birthx < 0.99:
        puff_sparsity = np.clip(np.random.uniform(low=self.birthx, high=1.0), 0.0, 1.0)
        drop_idxs = self.data_puffs['puff_number'].unique()
        drop_idxs = pd.Series(drop_idxs).sample(frac=(1.00-puff_sparsity))
        self.data_puffs = self.data_puffs.query("puff_number not in @drop_idxs") # No deep copy being made

    if self.diffusion_min < (self.diffusion_max - 0.01):
        diffx = np.random.uniform(low=self.diffusion_min, high=self.diffusion_max)
        self.diffusion_adjust(diffx)

    # Generalization: Randomly flip plume data across x_axis
    if self.flipping:
        self.flipx = -1.0 if np.random.uniform() > 0.5 else 1.0 
    else:
        self.flipx = 1.0
    # if self.flipx < 0:
    #     self.data_wind = self.data_wind_all.copy(deep=True)
    #     self.data_wind.loc[:,'wind_y'] *= self.flipx
    #     self.data_puffs = self.data_puffs.copy(deep=True)
    #     self.data_puffs.loc[:,'y'] *= self.flipx 
    #     # print(self.data_puffs.shape)
    # else:
    #     self.data_wind = self.data_wind_all

    self.data_wind = self.data_wind_all

    # Initialize agent to random location 
    # self.agent_location = self.get_initial_location(algo='quantile')
    self.agent_location = self.get_initial_location(self.loc_algo)
    self.agent_location_last = self.agent_location
    self.agent_location_init = self.agent_location

    self.stray_distance = self.get_stray_distance()
    self.stray_distance_last = self.stray_distance

    self.agent_angle = self.get_initial_angle(self.angle_algo)
    if self.verbose > 0:
      print("Agent initial location {} and orientation {}".format(self.agent_location, self.agent_angle))
    self.agent_velocity_last = np.array([0, 0])

    # self.wind_ground = self.get_current_wind_xy() # Observe after flip
    self.wind_ground = self.get_current_wind_xy() # Observe after flip
    if self.odor_scaling:
        self.odorx = np.random.uniform(low=0.5, high=1.5) # Odor generalize
    observation = self.sense_environment()
    if self.action_feedback:
        observation = np.concatenate([observation, np.zeros(2)])

    self.found_plume = True if observation[-1] > 0. else False 
    return observation


  def get_oob(self):
    # better restricted bounds    
    # bbox = {'x_min':-2, 'x_max':15, 'y_min':-5, 'y_max':5 }    
    # is_outofbounds = (self.agent_location[0] < bbox['x_min']) or \
    #                  (self.agent_location[0] > bbox['x_max']) or \
    #                  (self.agent_location[1] < bbox['y_min']) or \
    #                  (self.agent_location[1] > bbox['y_max']) 

    is_outofbounds = self.stray_distance > self.stray_max # how far agent can be from closest puff-center
    # if 'switch' in self.dataset: # large perturbations
    #     # bbox = {'x_min':-0.5, 'x_max':10, 'y_min':-3, 'y_max':3 }    

    return is_outofbounds

  def get_current_wind_xy(self):
    # df_idx = self.data_wind.query("time == {}".format(self.t_val)).index[0] # Safer
    df_idx = self.data_wind.query(f"tidx == {self.tidx}").index[0] # Safer
    return self.data_wind.loc[df_idx,['wind_x', 'wind_y']].tolist() # Safer

  # "Transition function"
  def step(self, action):
    """
    return observation, reward, done, info
    """
    self.episode_step += 1 
    self.agent_location_last = self.agent_location
    # Update internal variables
    try:
        self.tidx = self.tidxs[self.episode_step + self.step_offset]
        self.t_val = self.t_vals[self.episode_step + self.step_offset]
    except Exception as ex:
        # Debug case where the env tries to access t_val outside puff_data!
        print(ex, self.episode_step, self.step_offset, self.t_val_min, self.t_vals[-5:], self.tidxs[-5:])
        sys.exit(-1)
    
    self.stray_distance_last = self.stray_distance
    self.stray_distance = self.get_stray_distance()
    
    self.wind_ground = self.get_current_wind_xy()
    # print(self.wind_ground)

    # Unpack action
    if self.verbose > 1:
        print("step action:", action, action.shape)
    assert action.shape == (2,)
    if self.squash_action:
        action = (np.tanh(action) + 1)/2
    action = np.clip(action, 0.0, 1.0)
    move_action = action[0] # Typically between [0.0, 1.0]
    turn_action = action[1] # Typically between [0.0, 1.0]
    # print(action)

    # Action: Clip & self.rescale to support more algorithms
    # assert move_action >= 0 and move_action <= 1.0
    # assert turn_action >= 0 and turn_action <= 1.0
    if self.rescale:
        move_action = np.clip(move_action, -1.0, 1.0)
        move_action = (move_action + 1)/2 
        turn_action = np.clip(turn_action, -1.0, 1.0)
        turn_action = (turn_action + 1)/2 

    # Action noise (multiplicative)
    move_action *= 1.0 + np.random.uniform(-self.act_noise, +self.act_noise) 
    turn_action *= 1.0 + np.random.uniform(-self.act_noise, +self.act_noise) 

    if self.flipping and self.flipx < 0:
    	turn_action = 1 - turn_action

    # Turn/Update orientation and move to new location 
    old_angle_radians = np.angle(self.agent_angle[0] + 1j*self.agent_angle[1], deg=False)
    new_angle_radians = old_angle_radians + self.turn_capacity*self.turnx*(turn_action - 0.5)*self.dt # in radians
    self.agent_angle = [ np.cos(new_angle_radians), np.sin(new_angle_radians) ]    
    assert np.linalg.norm(self.agent_angle) < 1.1

    # New location = old location + agent movement + wind advection
    agent_move_x = self.agent_angle[0]*self.move_capacity*self.movex*move_action*self.dt
    agent_move_y = self.agent_angle[1]*self.move_capacity*self.movex*move_action*self.dt
    wind_drift_x = self.wind_ground[0]*self.dt
    wind_drift_y = self.wind_ground[1]*self.dt
    if self.walking:
        wind_drift_x = wind_drift_y = 0
    self.agent_location = [
      self.agent_location[0] + agent_move_x + wind_drift_x,
      self.agent_location[1] + agent_move_y + wind_drift_y,
    ]
    self.agent_velocity_last = np.array([agent_move_x, agent_move_y])/self.dt # For relative wind calc.

    ### ----------------- End conditions / Is the trial over ----------------- ### 
    is_home = np.linalg.norm(self.agent_location) <= self.homed_radius 
    is_outoftime = self.episode_step >= self.episode_steps_max - 1           
    is_outofbounds = self.get_oob()
    done = bool(is_home or is_outofbounds or is_outoftime)

    # Autocurricula
    # 0.999**1000 = 0.37
    # 0.998**1000 = 0.16
    # 0.997**1000 = 0.05
    # 0.996**1000 = 0.02
    # 0.995**1000 = 0.007
    # 0.99**400 = 0.02
    # 0.95**100 = 0.006
    if is_home and self.auto_movex:
        self.movex = 1 + 0.95*(self.movex - 1)
    if is_home and self.auto_reward:
        self.reward_decay *= 0.995

    # Observation
    observation = self.sense_environment()

    ### ----------------- Reward function ----------------- ### 
    reward = self.rewards['homed'] if is_home else self.rewards['tick']
    if observation[2] <= config.env['odor_threshold'] : # if off plume, more tick penalty
        reward += 5*self.rewards['tick']

    # Reward shaping         
    if is_outofbounds and 'oob' in self.r_shaping:
        # Going OOB should be worse than radial reward shaping
        # OOB Overshooting should be worse!
        oob_penalty = 5*np.linalg.norm(self.agent_location) + self.stray_distance
        oob_penalty *= 2 if self.agent_location[0] < 0 else 1  
        reward -= oob_penalty
         


    # Radial distance decrease at each STEP of episode
    r_radial_step = 0
    if 'step' in self.r_shaping:
        r_radial_step = 5*( np.linalg.norm(self.agent_location_last) - np.linalg.norm(self.agent_location) )
        r_radial_step = min(0, r_radial_step) if observation[2] <= config.env['odor_threshold'] else r_radial_step
        # Multiplier for overshooting source
        if 'overshoot' in self.r_shaping and self.agent_location[0] < 0:
            r_radial_step *= 2 # Both encourage and discourage agent more
        # Additive reward for reducing stray distance from plume
        if ('stray' in self.r_shaping) and (self.stray_distance > self.stray_max/3):
                r_radial_step += 1*(self.stray_distance_last - self.stray_distance)
        reward += r_radial_step * self.reward_decay


    # Walking agent: Metabolic cost: penalize forward movement
    r_metabolic = 0 # for logging
    if self.walking and 'metabolic' in self.r_shaping:
        delta_move = np.linalg.norm(np.array(self.agent_location_last) - np.array(self.agent_location))
        # r_metabolic = -5.*delta_move
        delta_move = 1 if delta_move > 0 else 0
        r_metabolic += self.rewards['tick']*delta_move
        reward += r_metabolic

    # Radial distance decrease at END of episode    
    radial_distance_reward = 0 # keep for logging
    if done and 'end' in self.r_shaping:
        # 1: Radial distance r_decreasease at end of episode
        radial_distance_decrease = ( np.linalg.norm(self.agent_location_init) - np.linalg.norm(self.agent_location) )
        # radial_distance_reward = radial_distance_decrease - np.linalg.norm(self.agent_location)
        # reward += radial_distance_reward 
        # reward -= np.linalg.norm(self.agent_location)
        # end_reward = -np.linalg.norm(self.agent_location)*(1+self.stray_distance) + radial_distance_decrease
        self.stray_distance = self.get_stray_distance()
        # end_reward = -2*self.stray_distance # scale to be comparable with sum_T(r_step)
        end_reward = radial_distance_decrease - self.stray_distance
        reward += end_reward

    r_location = 0 # incorrect, leads to cycling in place
    if 'loc' in self.r_shaping:
        r_location = 1/( 1  + np.linalg.norm(np.array(self.agent_location)) )
        r_location /= (1 + 5*self.stray_distance)
        reward += r_location 

    if 'turn' in self.r_shaping:
        reward -= 0.05*np.abs(2*(turn_action - 0.5))

    if 'move' in self.r_shaping:
        reward -= 0.05*np.abs(move_action)

    if 'found' in self.r_shaping:
        if self.found_plume is False and observation[-1] > 0.:
            # print("found_plume")
            reward += 10
            self.found_plume = True


    reward = reward*self.rewardx # Scale reward for A3C
    
    # Optional/debug info
    done_reason = "HOME" if is_home else \
        "OOB" if is_outofbounds else \
        "OOT" if is_outoftime else \
        "NA"    
    info = {
        't_val':self.t_val, 
        'tidx':self.tidx, 
        'flipx':self.flipx,
        'location':self.agent_location, 
        'location_last':self.agent_location_last, 
        'location_initial':self.agent_location_init, 
        'stray_distance': self.stray_distance,
        'wind_ground': self.wind_ground,
        'angle': self.agent_angle,
        'reward': reward,
        'r_radial_step': r_radial_step,
        # 'reward_decay': self.reward_decay,
        # 'r_radial_ep': radial_distance_reward,
        # 'r_metabolic': r_metabolic,
        'movex': self.movex,
        'done': done_reason if done else None,
        # 'outcomes': self.outcomes,
        'radiusx': self.radiusx,
        }

    if done:
        self.outcomes.append(done_reason)
        if len(self.outcomes) > 10:
            self.outcomes = self.outcomes[1:] # maintain list size

    if done and self.verbose > 0:
        print("{} at (x,y): {}, {} steps w/ reward {}".format( \
            done_reason, 
            self.agent_location, 
            self.episode_step, 
            reward))

    if self.action_feedback:
        # print(observation.shape, action.shape)
        observation = np.concatenate([observation, action])

    if self.flipping and self.flipx < 0:
    	observation[1] *= -1.0 # observation: [x, y, o] 
     
    self.episode_reward += reward
    
    if done:
        info['episode'] = {'r': self.episode_reward }


    if self.verbose > 0:
        print(observation, reward, done, info)
    return observation, reward, done, info

  def render(self, mode='console'):
    # raise NotImplementedError()
    return

  def close(self):
    del self.data_puffs_all
    del self.data_wind_all
    pass



#### 
class PlumeEnvironmentDiscreteActionWrapper(PlumeEnvironment):
    """
    TODO: Describe discrete actions
    """
    def __init__(self, dummy, **kwargs):
        self.venv = PlumeEnvironment(**kwargs)
        # Discrete action agent maintains current state: move-step and turn-step
        self.agent_movestep_DA = 0 # Used when discrete action space
        self.agent_turnstep_DA = 0 # Used when discrete action space

        self.action_space = spaces.MultiDiscrete([ [0,2], [0, 2] ])
        self.observation_space = self.venv.observation_space

    def step(self, action):
        # TODO - Discretize actions
        # print(action, type(action))
        # assert isinstance(action, tuple)
        delta_move = (action[0]-1)*self.move_capacity/2 # 0,1,2 --> -delta,0,+delta
        self.agent_movestep_DA = np.clip(self.agent_movestep_DA + delta_move, 0.0, self.move_capacity)  

        delta_turn = (action[1]-1)*1.0/8 # 0,1,2 --> -delta,0,+delta
        self.agent_turnstep_DA = self.agent_turnstep_DA + delta_turn  # Used when discrete action space
        self.agent_turnstep_DA = np.clip(self.agent_turnstep_DA, -1.0, 1.0)
        move_action = self.agent_movestep_DA
        turn_action = self.agent_turnstep_DA
        # print("turn_action, delta_turn", turn_action, delta_turn)
        # print("move_action, delta_move", move_action, delta_move)

        observations, rewards, dones, infos = self.venv.step(action)
        return observations, rewards, dones, infos

    def reset(self):
        obs = self.venv.reset()
        return obs

    def render(self, mode):
        self.venv.render(mode)

    def close(self):
        self.venv.close()


class PlumeFrameStackEnvironment(gym.Env):
    """
    Frame stacking wrapper for PlumeEnvironment
    Adapted from: https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/common/vec_env/vec_frame_stack.html#VecFrameStack
    """

    def __init__(self, n_stack, masking=None, stride=1, **kwargs):
        # mask in [None, 'odor', 'wind']
        # stride can be int or 'log'

        self.n_stack = n_stack
        self.masking = masking
        self.stride = stride

        self.venv = PlumeEnvironment(**kwargs)
        venv = self.venv
        wrapped_obs_space = venv.observation_space
        low = np.repeat(wrapped_obs_space.low, self.n_stack, axis=-1)
        high = np.repeat(wrapped_obs_space.high, self.n_stack, axis=-1)
        self.stackedobs = np.zeros(low.shape, low.dtype)
        # Parse stride-options
        if isinstance(self.stride, int):
            self.historyobs = np.zeros(wrapped_obs_space.shape[0] * self.n_stack * self.stride, wrapped_obs_space.dtype) # Full history
        if isinstance(self.stride, str) and 'log' in self.stride:
            self.historyobs = np.zeros(wrapped_obs_space.shape[0] * (1+2**(self.n_stack-1)), wrapped_obs_space.dtype) # Full history
        # print(self.historyobs.shape, self.stackedobs.shape)

        self.observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        self.action_space = self.venv.action_space

    def step(self, action):
        observations, rewards, dones, infos = self.venv.step(action)
        las = observations.shape[-1] # last_axis_size

        # Observation history are shifted left by las on each step
        self.historyobs = np.roll(self.historyobs, shift=-las, axis=-1)
        self.historyobs[..., -las:] = observations # load latest observation

        # self.historyobs --> self.stackedobs
        # Select observations to use to return to agent from history
        if isinstance(self.stride, int):
            idxs = ([0]+list(range(1, self.n_stack*self.stride, self.stride)))[:self.n_stack]
        if isinstance(self.stride, str) and 'log' in self.stride:
            idxs = ([0]+[2**i for i in range(0, self.n_stack)])[:self.n_stack]
        history_chronological = np.flip(self.historyobs) # simplify indexing
        for i in range(len(idxs)):
            self.stackedobs[i*las:(i+1)*las] = history_chronological[idxs[i]*las:(idxs[i]+1)*las]
        self.stackedobs = np.flip(self.stackedobs) # flip back

        # Masking & leave current/latest observation intact
        # Obs: [w_x, w_y, odor]
        if self.masking is not None and 'odor' in self.masking:
            mask = np.array([ 0 if ((i+1) % las == 0) else 1 for i in range(self.stackedobs.shape[0]) ])  
            mask[-las:] = 1
            self.stackedobs *= mask
        if self.masking is not None and 'wind' in self.masking:
            mask = np.array([ 1 if ((i+1) % las == 0) else 0 for i in range(self.stackedobs.shape[0]) ])  
            mask[-las:] = 1
            self.stackedobs *= mask

        return self.stackedobs, rewards, dones, infos

    def reset(self):
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs

    def render(self, mode):
        self.venv.render(mode)

    def close(self):
        self.venv.close()

# class PlumeFrameStackEnvironment1(gym.Env):
#     """
#     Frame stacking wrapper for PlumeEnvironment
#     Adapted from: https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/common/vec_env/vec_frame_stack.html#VecFrameStack
#     """

#     def __init__(self, n_stack, masking=None, **kwargs):
#         # mask in [None, 'odor', 'wind']

#         self.n_stack = n_stack
#         self.masking = masking

#         self.venv = PlumeEnvironment(**kwargs)
#         venv = self.venv
#         wrapped_obs_space = venv.observation_space
#         low = np.repeat(wrapped_obs_space.low, self.n_stack, axis=-1)
#         high = np.repeat(wrapped_obs_space.high, self.n_stack, axis=-1)
#         self.stackedobs = np.zeros(low.shape, low.dtype)
#         self.observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
#         self.action_space = self.venv.action_space

#     def step(self, action):
#         observations, rewards, dones, infos = self.venv.step(action)
#         las = observations.shape[-1] # last_ax_size
#         # Observations are shifted left by las on each step
#         self.stackedobs = np.roll(self.stackedobs, shift=-las, axis=-1)
#         self.stackedobs[..., -las:] = observations

#         # Masking & leave current/latest observation intact
#         # Obs: [w_x, w_y, odor]
#         if self.masking is not None and 'odor' in self.masking:
#             mask = np.array([ 0 if ((i+1) % 3 == 0) else 1 for i in range(self.stackedobs.shape[0]) ])  
#             mask[-las:] = 1
#             self.stackedobs *= mask
#         if self.masking is not None and 'wind' in self.masking:
#             mask = np.array([ 1 if ((i+1) % 3 == 0) else 0 for i in range(self.stackedobs.shape[0]) ])  
#             mask[-las:] = 1
#             self.stackedobs *= mask

#         return self.stackedobs, rewards, dones, infos

#     def reset(self):
#         obs = self.venv.reset()
#         self.stackedobs[...] = 0
#         self.stackedobs[..., -obs.shape[-1]:] = obs
#         return self.stackedobs

#     def render(self, mode):
#         self.venv.render(mode)

#     def close(self):
#         self.venv.close()

# class PlumeFrameStackEnvironment0(gym.Env):
#     """
#     Frame stacking wrapper for PlumeEnvironment
#     Adapted from: https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/common/vec_env/vec_frame_stack.html#VecFrameStack
#     """

#     def __init__(self, n_stack, **kwargs):
#         self.venv = PlumeEnvironment(**kwargs)
#         venv = self.venv
#         self.n_stack = n_stack
#         wrapped_obs_space = venv.observation_space
#         low = np.repeat(wrapped_obs_space.low, self.n_stack, axis=-1)
#         high = np.repeat(wrapped_obs_space.high, self.n_stack, axis=-1)
#         self.stackedobs = np.zeros(low.shape, low.dtype)
#         self.observation_space = spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
#         self.action_space = self.venv.action_space

#     def step(self, action):
#         observations, rewards, dones, infos = self.venv.step(action)
#         las = observations.shape[-1] # last_ax_size
#         self.stackedobs = np.roll(self.stackedobs, shift=-las, axis=-1)
#         self.stackedobs[..., -observations.shape[-1]:] = observations
#         return self.stackedobs, rewards, dones, infos

#     def reset(self):
#         obs = self.venv.reset()
#         self.stackedobs[...] = 0
#         self.stackedobs[..., -obs.shape[-1]:] = obs
#         return self.stackedobs

#     def render(self, mode):
#         self.venv.render(mode)

#     def close(self):
#         self.venv.close()