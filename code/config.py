import os
import sys
import glob
import numpy as np
import matplotlib

import socket
MACHINE = socket.gethostname().lower()
datadir = '/src/smartFlies/data/published_results/reproduce/'
if MACHINE == 'mycroft':
	datadir = '/data/users/satsingh/plumedata/'
if (MACHINE == 'salarian') or (MACHINE == 'cylon'):
	datadir = '/data1/users/satsingh/plumedata/'

seed_global = 137

traj_colormap = { 
	# 'on': 'lime',
	# 'on': 'darkgreen',
	'on': 'seagreen',
	# 'on': 'mediumseagreen',
	# 'on': 'royalblue',
	# 'on': 'dodgerblue',
	# 'on': 'blue',

	'off': 'blue',
	# 'off': 'dodgerblue', # lighter than royalblue
	# 'off': 'royalblue',
	# 'off': 'brown',
	# 'off': 'crimson',
	# 'off': 'red',
}

regime_colormap = {
					'SEARCH': 'red', 
					# 'SEARCH': 'brown', 

                   # 'TRACK':'darkolivegreen', # darker
                   # 'TRACK':'forestgreen', # standard green
                   'TRACK':'seagreen', # just right
                   # 'TRACK':'limegreen', 
                   
                   # 'RECOVER':'mediumslateblue', 	
                   'RECOVER':'slateblue', 
                   }
outcome_colormap = {'HOME': 'g', 
				    'OOB':'r', 
				    'OOT':'b'}

ttcs_colormap = {'HOME': 'b', 'OOB':'darkorange'}


plume_color = matplotlib.colors.to_rgba('gray')
# from sim_analysis.py
# rgba_colors[:,0:3] = matplotlib.colors.to_rgba('gray')[:3] # decent
# rgba_colors[:,0:3] = matplotlib.colors.to_rgba('darkgray')[:3] # decent
# rgba_colors[:,0:3] = matplotlib.colors.to_rgba('dimgray')[:3] # decent
# rgba_colors[:,0:3] = matplotlib.colors.to_rgba('darkslategray')[:3] # too dark
# rgba_colors[:,0:3] = matplotlib.colors.to_rgba('lightsteelblue')[:3] # ok
# rgba_colors[:,0:3] = matplotlib.colors.to_rgba('red')[:3] 
# rgba_colors[:,0:3] = matplotlib.colors.to_rgba('lightskyblue')[:3] 



# mwidth, mheight = 5.5, 9 # Manuscript usable dimensions for NeurIPS/ICLR
mwidth, mheight = 7, 9 # Manuscript usable dimensions for IEEE

# metadata associated with some seeds
seedmeta = {
	'2760377': {'recover_min':12, 'recover_max': 30, },
	# '3199993': {'recover_min':12, 'recover_max': 25, },
	'3307e9': {'recover_min':12, 'recover_max': 35, },
	'541058': {'recover_min':12, 'recover_max': 38, },
	# '9781ba': {'recover_min':12, 'recover_max': 25, },
}


env = {
	# 'rescale': False,
	# 'sim_steps_max': 300, 
	# 'reset_offset_tmax': 60.0 - 300.0/25, # t_val_min - sim_steps_max/fps
	# 'reset_offset_tmax': 25.00, # seconds
	# 'homed_radius': 0.2, # meters
	# 'stray_distance': 2.0, # meters
    'odor_threshold': 0.0001, # arbit units
    # 'odor_threshold': 1e-8, # arbit units
	'arena_bounds': {
		'x_min':-5, 
		'x_max':20, 
      	'y_min':-5, 
      	'y_max':5
      	},	

	# Max agent CW/CCW turn per second
	# 'turn_capacity': 25*np.pi * 0.75, 
	
	# Max agent speed in m/s
	# 'move_capacity': 2.5, 	
	# 'curriculum': True, # set in cli train
	# 'difficulty': 0.5, # Curriculum difficulty \in [0.0, 1.0]
	# 'difficulty': 0.65, # Curriculum difficulty \in [0.0, 1.0]
}
