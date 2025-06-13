import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import gym
import pandas as pd
import matplotlib.pyplot as plt

# import matplotlib.path as mpath
# import matplotlib.lines as mlines
# import matplotlib.patches as mpatches
# from matplotlib.collections import PatchCollection

RESCALE=False # Added to enable algorithms other than DDPG


# Random Agent
# https://github.com/openai/gym/blob/master/examples/agents/random_agent.py
# TODO: Buggy
class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        action = self.action_space.sample()
        # print(self.action_space, action)
        return np.array([action]) 


### Wrapper for SB policy networks ### 
class TrainedAgent(object):
    def __init__(self, model):
        self.model = model
    def act(self, observation, reward, done):
        action, _ = self.model.predict(observation, deterministic=True)
        return action


# Surge RANDOM hardcoded policy agent
class SurgeRandomAgent(object):
    """
    Surge: flow upwind when have odor
    Random: [0, pi/2] right turn otherwise
    """    
    def __init__(self, action_space):
        self.action_space = action_space
        self.mode = ['cast', 'surge'][1]
        # self.max_step = 0.25
        self.max_step = 1.0

    def act(self, observation, reward, done):
        # Decide mode
        odor_observation = observation[0][2]
        if odor_observation > 0:
            self.mode = 'surge'
            # self.cast_counter = self.cast_count_max
        else:
            self.mode = 'cast'

        # Decide turning
        if self.mode == 'surge':
            # In odor plume - turn/move towards it directly
            # print(observation.shape)
            wind_angle_relative = [observation[0][0], observation[0][1]]
            wind_relative_angle_radians = np.angle( wind_angle_relative[0] + 1j*wind_angle_relative[1], deg=False )
            if wind_relative_angle_radians < 0: # [-pi, pi] --> [0, 2*pi]
                wind_relative_angle_radians = 2*np.pi + wind_relative_angle_radians

            agent_turn = 0.5 + (wind_relative_angle_radians - np.pi)/(2*np.pi) 
            # agent_turn = wind_relative_angle_radians/(2*np.pi) # Interestingly, behaves the same as above
            # print("Turn radians/2*pi:", agent_turn)
            step_size = self.max_step
            action = np.array([step_size, agent_turn])

        else: # Random movement if not in either?
            # print('cast random')
            step_size = self.max_step
            action = np.array([step_size, np.random.uniform(0.0,0.5)])
            # Expected to only take Right turns
        action = np.expand_dims(action,axis=0) 

        if RESCALE:
            # Rescale actions [0, 1] --> [-1,+1]
            action = action*2 - 1

        return action


# Surge cast hardcoded policy agent
class SurgeCastAgent(object):
    """
    Surge-Cast definitions from: 
    Tracking Odor Plumes in a Laminar Wind Field with Bio-Inspired Algorithms
    Thomas Lochmatter and Alcherio Martinoli
    • Upwind surge: straight upwind movement as long as the moth is in the plume;
    • Casting: counter-turning (zig-zagging) to reacquire the plume right after loosing track of it;
    • Spiraling 1: an irregular, spiral-like movement to reacquire the plume if casting did not succeed.
    """       
    def __init__(self, action_space, ):
        self.action_space = action_space
        self.mode = 'cast' # Initially
        self.cast_count_max = 8
        self.cast_counter = self.cast_count_max
        self.wind_angle_odor = 0 # Last wind-angle when IN plume 

    def act(self, observation, reward, done):
        # Decide mode
        odor_observation = observation[0][2]
        if odor_observation > 0:
            self.mode = 'surge'
            self.cast_counter = self.cast_count_max
        else:
            self.mode = 'cast'
            self.cast_counter -= 1

        # Common: measure wind 
        wind_angle_relative = [observation[0][0], observation[0][1]]
        wind_relative_angle_radians = np.angle( wind_angle_relative[0] + 1j*wind_angle_relative[1], deg=False )
        if wind_relative_angle_radians < 0: # [-pi, pi] --> [0, 2*pi]
            wind_relative_angle_radians = 2*np.pi + wind_relative_angle_radians

        # Decide turning
        if self.mode == 'surge':
            # In odor plume - surge upwind
            agent_turn = 0.5 + (wind_relative_angle_radians - np.pi)/(2*np.pi) 
            # agent_turn = wind_relative_angle_radians/(2*np.pi) # Interestingly, behaves the same as above
            step_size = 1.0     
            action = np.array([step_size, agent_turn])
            self.wind_angle_odor = wind_relative_angle_radians
        else: 
            # Off odor plume, so "cast cross-wind"            
            # agent_turn = np.random.uniform(0.0,0.5) # Expected to only take Right turns
            if self.cast_counter >= self.cast_count_max/2:
                # Move cross-wind away from plume 
                agent_turn = 0.5 + -(wind_relative_angle_radians - np.pi/2)/(2*np.pi) 
            else:
                # Move in opposite directon of when you left plume
                agent_turn = 0.5 + (wind_relative_angle_radians - np.pi/2)/(2*np.pi) 

            step_size = 1.0
            action = np.array([step_size, agent_turn])            
        action = np.expand_dims(action,axis=0) 

        if RESCALE:
            # Rescale actions [0, 1] --> [-1,+1]
            action = action*2 - 1

        # print(self.mode, action)
        return action








class RandomTurnRadialRewardAgent(object):
    """
     Assumes radial reward shaping: 
     if last reward +ve, then keep angle at 0, else randomize angle. Move fixed 0.1 distance
    """
    def __init__(self, action_space):
        self.action_space = action_space
        self.last_reward = 0
#         self.step_size = 0.1

    def act(self, observation, reward, done):
        if self.last_reward > 0: # Good
            step_size = 0.4
            action = np.array([step_size, 0])
        else: # Bad
            step_size = 0.2
            action = np.array([step_size, np.random.uniform(0,0.75)])
        action = np.expand_dims(action,axis=0) 
        self.last_reward = reward

        if RESCALE:
            # Rescale actions [0, 1] --> [-1,+1]
            action = action*2 - 1

        return action




##### DISCRETE ACTION SPACE #####
class SurgeCastDiscreteActionAgent(object):
    """
    Surge-Cast definitions from: 
    Tracking Odor Plumes in a Laminar Wind Field with Bio-Inspired Algorithms
    Thomas Lochmatter and Alcherio Martinoli
    • Upwind surge: straight upwind movement as long as the moth is in the plume;
    • Casting: counter-turning (zig-zagging) to reacquire the plume right after loosing track of it;
    • Spiraling 1: an irregular, spiral-like movement to reacquire the plume if casting did not succeed.
    """
    
    def __init__(self, action_space):
        self.action_space = action_space
        self.mode = ['cast', 'surge'][1]

    def act(self, observation, reward, done):
        # Decide mode
        odor_observation = observation[0][2]
        if odor_observation > 0:
            self.mode = 'surge'
            # self.cast_counter = self.cast_count_max
        else:
            self.mode = 'cast'

        # Decide turning
        if self.mode == 'surge':
            # In odor plume - turn/move towards it directly
            # print(observation.shape)
            wind_angle_relative = [observation[0][0], observation[0][1]]
            wind_relative_angle_radians = np.angle( wind_angle_relative[0] + 1j*wind_angle_relative[1], deg=False )
            # if wind_relative_angle_radians < 0: # [-pi, pi] --> [0, 2*pi]
            #     wind_relative_angle_radians = 2*np.pi + wind_relative_angle_radians

            # agent_turn = np.sign(0.5 + (wind_relative_angle_radians - np.pi)/(2*np.pi)) + 1
            # agent_turn = np.random.choice([agent_turn, 1], p=[0.9, 0.1])
            # agent_turn = 2 if wind_relative_angle_radians < np.pi/2 else 0
            # agent_turn = -1*np.sign(0.5 + (wind_relative_angle_radians - np.pi)/(2*np.pi)) + 1
            # agent_turn_01 = 0.5 + (wind_relative_angle_radians - np.pi)/(2*np.pi) # 0-1
            wind_relative_angle_01 = wind_relative_angle_radians/(2*np.pi) # 0-1
            # if wind_relative_angle_01 < -0.25:
            #     agent_turn = 0
            # elif wind_relative_angle_01 > +0.25:
            #     agent_turn = 2
            # else:
            #     agent_turn = 0
            if wind_relative_angle_01 > -0.25 and wind_relative_angle_01 < +0.25:
                agent_turn = 2
            else:
                agent_turn = 2 

            print("surge: wind_rel_angle_01, agent_turn", wind_relative_angle_01, agent_turn)
 
            # agent_turn = np.random.choice([0, 1, 2])     
            step_size = 2     
            # step_size = np.random.choice([0, 1, 2])     
            # action = np.array([step_size, agent_turn])
            action = (step_size, agent_turn)

        else: # Random movement if not in either?
            # print('cast random')
            step_size = 2
            # action = np.array([step_size, np.random.choice([0, 2])])
            # Expected to only take Right turns
            agent_turn = np.random.choice([0, 1, 2], p=[0.4, 0.2, 0.4])
            action = (step_size, agent_turn)
            print("cast: agent_turn (random)", agent_turn)

        return np.array([action])
