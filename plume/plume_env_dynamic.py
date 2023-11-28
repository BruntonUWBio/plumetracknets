import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import glob

import pandas as pd
import numpy as np
import gym
from gym import spaces

import sim_analysis
import tqdm
from pprint import pprint

import config
from scipy.spatial.distance import cdist 
import sim_utils

class DynamicPlume:
  def __init__(self, 
        sim_dt=0.01,
        birth_rate=1.0,
        env_dt=0.04,
        birthx=1.0, # per-episode puff birth rate sparsity minimum
        birthx_max=1.0, # overall odor puff birth rate sparsity max
        wind_speed=0.5,
        wind_y_var=0.5,
        qvar=0.0, # Variance of init. location; higher = more off-plume initializations
        diff_max=0.8, # teacher curriculum
        diff_min=0.4, # teacher curriculum
        warmup=25, # warmup upto these many steps 
        max_steps=300, # max steps in episode (used for switch_idxs, ok to run longer)
        dataset=None, # optional: imitate a "dataset"
        verbose=0):
      super(DynamicPlume, self).__init__()

      # Init 
      # print(os.getcwd())
      self.verbose = verbose
      self.warmup = warmup
      self.max_steps = max_steps
      self.snapshots = self.init_snapshots(config.datadir)
      self.birthx = birthx
      self.steps_per_env_dt = 4 # env_dt/sim_dt hardcoded
      self.birth_rate = birth_rate
      self.wind_y_var = wind_y_var
      self.wind_speed = wind_speed
      self.wind_degree = 0
      # self.switch_counts = [0]*6 + [i for i in range(1, 13)] # mix of constant & switch
      self.switch_counts = [0, 0, 0, 1, 1, 1, 2, 4, 6, 8] # mix of constant & switch
      if dataset is not None and 'constant' in dataset:
        self.switch_counts = [0]
      if dataset is not None and 'noisy' in dataset:
        self.switch_counts = [0, 0, 1, 1, 1, 2, 3, 4, 5, 6] # mix of constant & switch

      self.diff_min = diff_min
      self.diff_max = diff_max
      self.qvar = qvar
      self.reset()

  def init_snapshots(self, snapshots_dir):
    fnames = list(glob.glob(f"{snapshots_dir}/*_snapshot.csv"))[:10]
    if len(fnames) < 1:
        print(len(fnames), snapshots_dir)
    return [ pd.read_csv(x) for x in fnames ]

  def sparsify(self, puff_df, birthx=1.0):
    keep_idxs = puff_df['puff_number'].sample(frac=np.clip(birthx, 0.0, 1.0))
    return puff_df.query("puff_number in @keep_idxs")

  def reset(self):
    self.ep_step = 0
    self.wind = [0.5, 0.0]
    self.wind_y_varx = np.random.uniform(low=0.8, high=1.2) # some randomness to how spread out the wind puffs will be
    self.puffs = self.snapshots[ np.random.randint(0, len(self.snapshots)) ].copy(deep=True)
    if np.random.uniform(0.0, 1.0) > 0.5: # Random flip
        self.puffs.loc[:,'y'] *= 1
    self.tidx = self.puffs['tidx'].unique().item()
    self.switches_ep = np.random.choice(self.switch_counts)
    self.switch_idxs = [] if self.switches_ep == 0 else np.random.randint(0, self.max_steps, self.switches_ep).tolist()
    if self.switches_ep in [1, 2]:
        self.switch_idxs = np.random.randint(0, int(self.max_steps/3), self.switches_ep).tolist()
    # self.switch_p = self.switches_ep/self.max_steps
    # Dynamic birthx for each episode
    if self.switches_ep == 0:    
        self.birthx_ep = np.random.uniform(low=self.birthx, high=1.0)
    else:
        self.birthx_ep = np.random.uniform(low=0.7, high=1.0)
    if self.birthx_ep < 0.95:
      self.puffs = self.sparsify(self.puffs, self.birthx_ep)
    # Warmup
    for i in range(np.random.randint(0, self.warmup)):
        self.step()

  def step(self):
    self.ep_step += 1
    # update puffs
    wind_t = pd.Series({'wind_x': self.wind[0], 'wind_y': self.wind[1], 'time':(self.tidx+1)/100})
    for i in range(self.steps_per_env_dt):
        self.tidx += 1
        self.puffs = sim_utils.manual_integrator(
            self.puffs[['puff_number', 'time', 'tidx', 'x', 'y', 'radius']], 
            wind_t, 
            self.tidx, 
            birth_rate=self.birth_rate*self.birthx_ep, 
            wind_y_var=self.wind_y_var*self.wind_y_varx)
    # update wind
    # if self.switches_ep > 0 and np.random.uniform(low=0.0, high=1.0) <= self.switch_p:
    if self.switches_ep > 0 and self.ep_step in self.switch_idxs:
        self.wind_degree = np.random.normal(0, 60)
        self.wind_degree = np.clip(self.wind_degree, -60, 60 )
        # self.wind_degree = np.random.uniform(-60, 60)
        wind_x = np.cos( self.wind_degree * np.pi / 180. )*self.wind_speed
        wind_y = np.sin( self.wind_degree * np.pi / 180. )*self.wind_speed
        if self.verbose > 0:
            print(f"tidx: {self.tidx} - wind_degree:{self.wind_degree}")
        self.wind = [wind_x, wind_y]


  def get_abunchofpuffs(self, max_samples=300):  
    # Z = self.puffs.query(f"tidx == {self.tidx}").loc[:,['x','y']]
    Z = self.puffs.loc[:,['x','y']]
    Z = Z.sample(n=max_samples, replace=False) if Z.shape[0] > max_samples else Z
    return Z

  def get_stray_distance(self, agent_location, max_samples=2000):
    loc_x = agent_location[0]
    loc_x_window = 1.0 # meters
    # y_median = self.puffs.query("(x >= (@loc_x - @loc_x_window)) and (x <= (@loc_x + @loc_x_window))")['y'].median()
    y_median = self.puffs.query("(x >= (@loc_x - @loc_x_window)) and (x <= (@loc_x + @loc_x_window))")['y'].mean()
    y_median = 0 if np.isnan(y_median) else y_median
    ystray = np.abs(agent_location[1] - y_median)
    return ystray
    # Z = self.get_abunchofpuffs(max_samples=max_samples)
    # Y = cdist(Z.to_numpy(), np.expand_dims(agent_location,axis=0), metric='euclidean')
    # try:
    #     minY = min(Y) 
    # except Exception as ex:
    #     print(f"Exception: {ex}, t:{self.t_val:.2f}, tidx:{self.tidx}, {Z}")  
    #     minY = np.array([0])      
    # return minY[0] # return float not float-array


  def get_current_wind_xy(self):
    return self.wind

  def get_initial_location(self, loc_algo):
    # assume quantile loc_algo
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
    loc_xy = np.array([X_mean + varx*X_var*np.random.randn(), 
        Y_mean + varx*Y_var*np.random.randn()]) 
    return loc_xy

  def get_concentration(self, x_val, y_val, min_radius=0.01, extent=0.0):
    if 'concentration' not in self.puffs.columns:
        self.puffs['x_minus_radius'] = self.puffs.x - self.puffs.radius
        self.puffs['x_plus_radius'] = self.puffs.x + self.puffs.radius
        self.puffs['y_minus_radius'] = self.puffs.y - self.puffs.radius
        self.puffs['y_plus_radius'] = self.puffs.y + self.puffs.radius
        self.puffs['concentration'] = (min_radius/self.puffs.radius)**3

    # xval_ext = xval_ext
    qx = "@x_val > x_minus_radius and @x_val < x_plus_radius"
    qy = "@y_val > y_minus_radius and @y_val < y_plus_radius"
    q = qx + ' and ' + qy
    d = self.puffs.query(q)
    return d.concentration.sum()

class PlumeEnvironment(gym.Env):
  """
  Documentation: https://gym.openai.com/docs/#environments
  Plume tracking
  """
  def __init__(self, 
    t_val_min=60.00, 
    sim_steps_max=300, # steps
    reset_offset_tmax=30, # seconds; max secs for initial offset from t_val_min
    dataset=None,
    move_capacity=2.5, # Max agent speed in m/s
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
    diff_max=0.8, # teacher curriculum
    diff_min=0.4, # teacher curriculum
    r_shaping=['step'], # 'step', 'end'
    rewardx=1.0, # scale reward for e.g. A3C
    rescale=False, # rescale/normalize input/outputs [redundant?]
    squash_action=False, # apply tanh and rescale (useful with PPO)
    walking=False,
    walk_move=0.05, # m/s (x100 for cm/s)
    walk_turn=1.0*np.pi, # radians/sec
    radiusx=1.0, 
    action_feedback=False,
    flipping=False, # Generalization/reduce training data bias
    odor_scaling=False, # Generalization/reduce training data bias
    obs_noise=0.0, # Multiplicative: Wind & Odor observation noise.
    act_noise=0.0, # Multiplicative: Move & Turn action noise.
    seed=137,
    verbose=0):
    super(PlumeEnvironment, self).__init__()

    self.arguments = locals()
    print("PlumeEnvironment:", self.arguments)

    np.random.seed(seed)    
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
    self.t_val_min = t_val_min
    self.episode_steps_max = sim_steps_max # Short training episodes to gather rewards
    self.t_val_max = self.t_val_min + self.reset_offset_tmax + 1.0*self.episode_steps_max/self.fps + 1.00


    # Other initializations -- many redundant, see .reset() 
    # self.agent_location = np.array([1, 0]) # TODO: Smarter
    self.agent_location = None
    self.agent_location_last = self.agent_location
    self.agent_location_init = self.agent_location
    random_angle = np.pi * np.random.uniform(0, 2)
    self.agent_angle_radians = [np.cos(random_angle), np.sin(random_angle)] # Sin and Cos of angle of orientation
    self.step_offset = 0 # random offset per trial in reset()
    self.t_val = 0.0
    self.tidx = 0
    self.tidx_min_episode = self.tidx
    self.tidx_max_episode = self.tidx
    self.wind_ground = 0.
    self.odor_ground = 0.
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
      'tick': -1/self.episode_steps_max,
      'homed': 101.0,
      }

    # dynamic plume
    self.dynamic = DynamicPlume(
        env_dt=self.dt,
        birthx=self.birthx, # per-episode puff birth rate sparsity minimum
        birthx_max=self.birthx_max, # overall odor puff birth rate sparsity max
        qvar=self.qvar, # Variance of init. location; higher = more off-plume initializations
        diff_max=self.diff_max, # teacher curriculum
        diff_min=self.diff_min, # teacher curriculum
        dataset=dataset,
        )



    # Define action and observation spaces
    # Actions:
    # Move [0, 1], with 0.0 = no movement
    # Turn [0, 1], with 0.5 = no turn... maybe change to [-1, 1]
    self.action_space = spaces.Box(low=0, high=+1, shape=(2,), dtype=np.float32)
    # Observations
    # Wind velocity [-1, 1] * 2, Odor concentration [0, 1]
    obs_dim = 3 if not self.action_feedback else 3+2
    self.observation_space = spaces.Box(low=-1, high=+1,
                                shape=(obs_dim,), dtype=np.float32)

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

    # Odor
    self.odor_ground = self.dynamic.get_concentration(self.agent_location[0], self.agent_location[1])
    odor_observation = self.odor_ground 
    if self.verbose > 1:
        print('odor_observation', odor_observation)
    if self.odor_scaling:
        odor_observation *= self.odorx # Random scaling to improve generalization 
    odor_observation *= 1.0 + np.random.uniform(-self.obs_noise, +self.obs_noise) # Add observation noise

    odor_observation = 0.0 if odor_observation < config.env['odor_threshold'] else odor_observation
    odor_observation = np.clip(odor_observation, 0.0, 1.0) # clip for neural net stability

    # Return
    observation = np.array(wind_observation + [odor_observation]).astype(np.float32) # per Gym spec
    if self.verbose > 1:
        print('observation', observation)
    return observation

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
        assert False
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

  def reset(self):
    """
    return Gym.Observation
    """
    self.dynamic.reset()
    self.episode_reward = 0
    self.episode_step = 0 # skip_steps already done during loading
    # Add randomness to start time PER TRIAL!

    # Initialize agent to random location 

    self.agent_location = self.dynamic.get_initial_location(self.loc_algo)
    # if self.loc_algo == 'quantile' else self.get_initial_location(self.loc_algo)
    self.agent_location_last = self.agent_location
    self.agent_location_init = self.agent_location

    self.stray_distance = self.dynamic.get_stray_distance(self.agent_location)
    self.stray_distance_last = self.stray_distance

    self.agent_angle = self.get_initial_angle(self.angle_algo)
    if self.verbose > 0:
      print("Agent initial location {} and orientation {}".format(self.agent_location, self.agent_angle))
    self.agent_velocity_last = np.array([0, 0])

    # self.wind_ground = self.get_current_wind_xy() # Observe after flip
    self.wind_ground = self.dynamic.get_current_wind_xy() # Observe after flip
    if self.odor_scaling:
        self.odorx = np.random.uniform(low=0.5, high=1.5) # Odor generalize
    observation = self.sense_environment()
    if self.action_feedback:
        observation = np.concatenate([observation, np.zeros(2)])

    self.found_plume = True if observation[-1] > 0. else False 
    return observation


  def get_oob(self):
    is_outofbounds = self.stray_distance > self.stray_max # how far agent can be from closest puff-center
    return is_outofbounds

  # "Transition function"
  def step(self, action):
    """
    return observation, reward, done, info
    """
    self.episode_step += 1 
    self.agent_location_last = self.agent_location
    # Update internal variables
    self.stray_distance_last = self.stray_distance
    self.stray_distance = self.dynamic.get_stray_distance(self.agent_location)
    
    self.wind_ground = self.dynamic.get_current_wind_xy()

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

    # Action noise (multiplicative)
    move_action *= 1.0 + np.random.uniform(-self.act_noise, +self.act_noise) 
    turn_action *= 1.0 + np.random.uniform(-self.act_noise, +self.act_noise) 

    # Action: Clip & self.rescale to support more algorithms
    # Turn/Update orientation and move to new location 
    old_angle_radians = np.angle( self.agent_angle[0] + 1j*self.agent_angle[1], deg=False )
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

    # Observation
    observation = self.sense_environment()

    ### ----------------- Reward function ----------------- ### 
    reward = self.rewards['homed'] if is_home else self.rewards['tick']
    if observation[2] <= config.env['odor_threshold'] : # if off plume, more tick penalty
        reward += 5*self.rewards['tick']

    # Reward shaping         
    if is_outofbounds and 'oob_fixed' in self.r_shaping:
        # Going OOB should be worse than radial reward shaping
        # OOB Overshooting should be worse!
        oob_penalty = 10
        # oob_penalty *= 2 if self.agent_location[0] < 0 else 1  
        reward -= oob_penalty

    if is_outofbounds and 'oob_loc' in self.r_shaping:
        oob_penalty = 5*np.linalg.norm(self.agent_location) + 5*self.stray_distance
        # oob_penalty *= 2 if self.agent_location[0] < 0 else 1  
        reward -= oob_penalty

             
    # X distance decrease at each STEP of episode
    r_xstep = 0
    if 'xstep' in self.r_shaping:
        r_xstep = 5*(self.agent_location_last[0] - self.agent_location[0])
        r_xstep = min(0, r_xstep) if observation[2] <= config.env['odor_threshold'] else r_xstep
        # Multiplier for overshooting source
        # if ('stray' in self.r_shaping) and (self.stray_distance > self.stray_max/3):
        #         r_xstep += 2.5*(self.stray_distance_last - self.stray_distance)
        reward += r_xstep * self.reward_decay

    # New Stray rewards/penalties
    # Additive reward for reducing stray distance from plume
    if ('stray_delta' in self.r_shaping) and (self.stray_distance > self.stray_max/4):
        reward += 1*(self.stray_distance_last - self.stray_distance) # higher when stray reducing 
    if ('stray_abs' in self.r_shaping) and (self.stray_distance > self.stray_max/4):
        # print("r_xstep, stray_abs: ", r_xstep, -0.1*self.stray_distance)
        reward += -0.05*self.stray_distance # 

    # Y-stray decrease at each STEP of episode
    r_ystray = 0
    # if 'ystray' in self.r_shaping:
    #     r_ystray = -0.1*ystray
    #     # print(reward, r_ystray, y_median, self.agent_location[1])
    #     reward += r_ystray * self.reward_decay


    # Radial distance decrease at each STEP of episode
    r_radial_step = 0
    if 'step' in self.r_shaping:
        r_radial_step = 5*( np.linalg.norm(self.agent_location_last) - np.linalg.norm(self.agent_location) )
        # Multiplier for overshooting source
        # if 'overshoot' in self.r_shaping and self.agent_location[0] < 0:
        #     r_radial_step *= 2 # Both encourage and discourage agent more
        reward += r_radial_step * self.reward_decay

    if 'step_pos' in self.r_shaping:
        r_radial_step = 5*( np.linalg.norm(self.agent_location_last) - np.linalg.norm(self.agent_location) )
        r_radial_step = min(0, r_radial_step) if observation[2] <= config.env['odor_threshold'] else r_radial_step
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
    if done and 'end_pos' in self.r_shaping:
        reward += 12/(1+np.linalg.norm(self.agent_location)) if observation[2] > config.env['odor_threshold'] else 0

    if done and 'end' in self.r_shaping:
        reward -= np.linalg.norm(self.agent_location)
        # reward -= np.linalg.norm(self.agent_location) + self.stray_distance
        # 1: Radial distance r_decreasease at end of episode
        # radial_distance_decrease = ( np.linalg.norm(self.agent_location_init) - np.linalg.norm(self.agent_location) )
        # radial_distance_reward = radial_distance_decrease - np.linalg.norm(self.agent_location)
        # reward += radial_distance_reward 
        # reward -= np.linalg.norm(self.agent_location)
        # end_reward = -np.linalg.norm(self.agent_location)*(1+self.stray_distance) + radial_distance_decrease
        # self.stray_distance = self.dynamic.get_stray_distance(self.agent_location, max_samples=3000) # Highest quality
        # end_reward = -2*self.stray_distance # scale to be comparable with sum_T(r_step)
        # end_reward = radial_distance_decrease - self.stray_distance
        # reward += 5*(end_reward)

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

    if is_home and self.auto_reward:
        self.dynamic.diff_min *= 1.01
        self.dynamic.diff_max *= 1.01
        self.dynamic.diff_min = min(self.dynamic.diff_min, 0.4)
        self.dynamic.diff_max = min(self.dynamic.diff_max, 0.8)

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
        'r_dict': {
            'r_radial_step': r_radial_step,
            'r_xstep': r_xstep,
            # 'r_ystray': r_ystray,
            },
        'movex': self.movex,
        'done': done_reason if done else None,
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
    pass



#### 
# class PlumeEnvironmentDiscreteActionWrapper(PlumeEnvironment):
#     """
#     TODO: Describe discrete actions
#     """
#     def __init__(self, dummy, **kwargs):
#         self.venv = PlumeEnvironment(**kwargs)
#         # Discrete action agent maintains current state: move-step and turn-step
#         self.agent_movestep_DA = 0 # Used when discrete action space
#         self.agent_turnstep_DA = 0 # Used when discrete action space

#         self.action_space = spaces.MultiDiscrete([ [0,2], [0, 2] ])
#         self.observation_space = self.venv.observation_space

#     def step(self, action):
#         # TODO - Discretize actions
#         # print(action, type(action))
#         # assert isinstance(action, tuple)
#         delta_move = (action[0]-1)*self.move_capacity/2 # 0,1,2 --> -delta,0,+delta
#         self.agent_movestep_DA = np.clip(self.agent_movestep_DA + delta_move, 0.0, self.move_capacity)  

#         delta_turn = (action[1]-1)*1.0/8 # 0,1,2 --> -delta,0,+delta
#         self.agent_turnstep_DA = self.agent_turnstep_DA + delta_turn  # Used when discrete action space
#         self.agent_turnstep_DA = np.clip(self.agent_turnstep_DA, -1.0, 1.0)
#         move_action = self.agent_movestep_DA
#         turn_action = self.agent_turnstep_DA
#         # print("turn_action, delta_turn", turn_action, delta_turn)
#         # print("move_action, delta_move", move_action, delta_move)

#         observations, rewards, dones, infos = self.venv.step(action)
#         return observations, rewards, dones, infos

#     def reset(self):
#         obs = self.venv.reset()
#         return obs

#     def render(self, mode):
#         self.venv.render(mode)

#     def close(self):
#         self.venv.close()


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