import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import numpy as np
import matplotlib.pyplot as plt
import pandas
import scipy
import scipy as sp
import scipy.stats
from scipy.integrate import odeint
import multiprocessing
import time
import pickle
import multiprocessing
import numpy as np
import scipy.stats
import time
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import os
import tqdm
import config
np.random.seed(config.seed_global)


# from config import sim as simc
from numba import jit # TODO


# wind/plume sim parameters
# dt = simc['dt']
# wind_magnitude = simc['wind_magnitude']
# wind_noisy = simc['wind_noisy'] # used by Inflexible/Floris generator
# birth_rate = simc['birth_rate']


def get_wind_vectors_flexible(T, wind_magnitude, local_state=None, regime=None):
    '''
    T: 1D array of timestamps
    local_state: need this to get unique noise arrays if running code in parallel
    '''
    if local_state is None:
        local_state = np.random.RandomState(0)

    # Setup baseline wind vectors: L --> R with 
    wind_degrees = np.zeros(len(T))
    wind_speeds = np.ones(len(T))*wind_magnitude #+ local_state.normal(0, 0.01*wind_magnitude, len(T))

    # 45 degree perturbation midway
    if 'switch' in regime:
        # TODO: Change this to happen at a certain fixed time
        # Since this is testing data, no need to randomize this time
        how_much = int(regime.replace('switch',''))
        wind_degrees_perturb = np.ones(len(T))*how_much
        wind_degrees_perturb[:int(len(T)/2)] = 0
        wind_degrees += wind_degrees_perturb

    # # Add random noise to wind speed
    # if 'noisy_speed' in features:
    #     wind_speeds += local_state.normal(0, 0.2*wind_magnitude, len(T))



    # Add random noise to wind degree
    if 'noisy' in regime:
        # every timestep noisy
        # noise = local_state.normal(0, 5, len(T)) 

        # switch every repN timesteps, with variance var degrees 
        # if 'noisy1' in regime:
        #     repN, var = 25, 5 # Small, fast changes
        # if 'noisy2' in regime:
        #     repN, var = 100, 45 # fewer switches
        # if 'noisy3' in regime:
        #     repN, var = 50, 45 # larger waves
        # if 'noisy4' in regime:
        #     repN, var = 100, 45 #
        # if 'noisy5' in regime:
        #     repN, var = 150, 30 #
        # if 'noisy6' in regime:
        #     repN, var = 300, 35 #
        # noise = local_state.normal(0, var, int(len(T)/repN)+1)
        # noise = np.repeat(noise, repN+1)[:len(T)]

        noise = np.zeros(len(T)) # Init
        repN = 100 # timesteps
        repN = 200 if 'noisy2' in regime else repN
        repN = 300 if 'noisy3' in regime else repN
        repN = 400 if 'noisy4' in regime else repN
        repN = 500 if 'noisy5' in regime else repN
        repN = 600 if 'noisy6' in regime else repN
        degz = 60 # +/- degz 

        # More evenly spaced
        switch_idxs = np.arange(len(T), step=repN, dtype=int)        
        switch_idxs = [ s + local_state.choice(np.arange(-int(repN/10), int(repN/10), dtype=int)) for s in switch_idxs ]
        switch_idxs = np.sort(switch_idxs)
        for idx in switch_idxs:
            noise[idx:] = local_state.normal(0, degz/2)

        # Random spacing
        # switch_N = int(len(T)/repN) # how many switches?
        # switch_idxs = np.random.choice(np.arange(len(T), step=50, dtype=int), size=switch_N, replace=False)        
        # switch_idxs = np.sort(switch_idxs)
        # print(switch_N, switch_idxs)
        # for idx in switch_idxs:
        #     # noise[idx:] = np.random.uniform(-degz, degz)        
        #     noise[idx:] = np.random.normal(0, degz/2)

        # limit max
        noise = np.clip(noise, -degz, degz)        

        wind_degrees += noise # no smoothing

        # smooth out additive noise        
        # noise_smooth = pd.Series(noise).rolling(50, win_type='triang').mean()
        # noise_smooth = noise_smooth.interpolate(limit_direction='backward') # Eliminate NaNs
        # wind_degrees += noise_smooth

    # if 'const2noisy' in regime:
    # 	# Assume that wind_degrees has already been processed for noisyX from above
    # 	# Then set the first 70 secs to 0 angle
    #     wind_degrees[:int(len(T)*(70/120))] = 0

    # Convert to X Y
    wind_x = np.cos( wind_degrees * np.pi / 180. )*wind_speeds
    wind_y = np.sin( wind_degrees * np.pi / 180. )*wind_speeds

    return wind_x, wind_y


def get_wind_xyt(duration, dt, wind_magnitude, verbose=True, regime='noisy3'):
    T = np.arange(0, duration, dt).astype('float64')
    if verbose:
        print("Generate and save wind data ... ")

    # if not flexible:
    #     wind_x, wind_y = get_wind_vectors_original(T, 
    #         wind_magnitude=wind_magnitude, 
    #         noisy=wind_noisy, 
    #         switch_direction=switch_direction)

    # if flexible:
    wind_x, wind_y = get_wind_vectors_flexible(T, wind_magnitude, regime=regime)

    data_wind = pd.DataFrame({'wind_x': wind_x[0:len(T)],
                              'wind_y': wind_y[0:len(T)],
                              'time': T})

    # Compress numerical data!
    n_times_pre = len(data_wind['time'].unique())
    for col in data_wind.select_dtypes(include='float').columns:
        if 'time' in col:
            data_wind[col] = data_wind[col].astype('float64') 
        else:    
            data_wind[col] = data_wind[col].astype('float16') # Use lightest floats
    assert n_times_pre == len(data_wind['time'].unique()) # Make sure quantization is lossless

    return data_wind


# Puff Simulation ODEINT
def puff_wind_diff_eq(xyr, t, *args):
    dt, wind_x, wind_y = args
    x,y,r = xyr

    idx = int(t/dt)
    if idx > len(wind_x)-1:
        idx = -1
    xdot = wind_x[idx]
    ydot = wind_y[idx]

    rdot = 0.01

    return [xdot, ydot, rdot]   

# Serial (not parallel) version
def integrate_puff_from_birth(args):
    T, wind_x, wind_y, birth_index, seed = args
    # Simulate once

    xyr_0 = [0,0,0.01] # initial x, y, radius
    dt = 0.01
    # wind_x, wind_y = get_wind_vectors_original(T, 
    #     local_state=local_state, 
    #     wind_magnitude=wind_magnitude, 
    #     switch_direction=switch_direction)

    # # Add some y-direction variation per puff
    # local_state = np.random.RandomState(seed)
    # wind_y_var = local_state.normal(0, wind_magnitude, len(T))

    vals, extra = odeint(puff_wind_diff_eq, xyr_0, T[birth_index:], 
                         args=(dt, wind_x, wind_y), 
                         full_output=True)
    Z = np.zeros([birth_index, 3])
    return np.vstack((Z, vals))

def get_puffs_raw(T, wind_x, wind_y, birth_rate, ncores=2, verbose=True):
    #### 
    print("Generating indices of plume puff births...")
    births = scipy.stats.poisson.rvs(birth_rate, size=len(T))
    birth_indices = []
    for idx, birth in enumerate(births):
        birth_indices.extend([idx]*birth)

    #### 
    print("Starting parallel simulation ({} cores) for each plume puff...".format(ncores))
    # Setup inputs for parallel simulation
    seeds = np.arange(0, len(birth_indices))
    inputs = [[T,
        wind_x,
        wind_y, 
        birth_indices[i], 
        seeds[i]] for i in range(0, len(birth_indices))]

    t_start = time.time()

    pool = multiprocessing.Pool(ncores)
    puffs = pool.map(integrate_puff_from_birth, inputs)
    comp_time = time.time() - t_start
    if verbose:
        print('Computation time: ', comp_time)
        print('ncores: ', ncores, ' puffs: ', len(puffs), ' steps: ', len(T))
        print('Computation time (C) per core (n) per puff (p) per step (s) = (C*n/p*s) = ', ncores*comp_time/(len(puffs)*len(T)))
        print('Total time = C*n/p*s')

    puffs_arr = np.stack(puffs)

    if save:
        if verbose:
            print("Save raw puff data...")
        data = {'puffs': puffs_arr, 'wind_x': wind_x, 'wind_y': wind_y, 'time': T}
        with open('puff_data_array.pickle', 'wb') as pickle_file:
            pickle.dump(data, pickle_file)

    return data

##### Parallel DF version #####
def integrate_puff_from_birth_df(args):
    T, tidxs, wind_x, wind_y, wind_y_var, birth_index, puff_index, seed = args

    xyr_0 = [0,0,0.01] # initial x, y, radius
    dt = 0.01

    # Add some y-direction variation per puff
    local_state = np.random.RandomState(seed)
    wind_dy = local_state.normal(0, wind_y_var, len(wind_y))


    vals, extra = odeint(puff_wind_diff_eq, xyr_0, T[birth_index:], 
                         args=(dt, wind_x, wind_y + wind_dy), 
                         full_output=True)

    puff_df = pd.DataFrame({
        'puff_number': puff_index, 
        'time': T[birth_index:], 
        'tidx': tidxs[birth_index:], 
        'x': vals[:, 0], 
        'y': vals[:, 1], 
        'radius': vals[:, 2],
        })

    # Postprocessing 
    puff_df = puff_df.query("(radius != 0) & (x<10) & (y<10) & (x>-2) & (y>-10)")

    return puff_df

def get_puffs_df_oneshot(wind_df, wind_y_var, birth_rate, ncores=2, verbose=True):
    T = wind_df['time'].to_numpy()
    tidxs = wind_df['tidx'].to_numpy()
    wind_x = wind_df['wind_x'].to_numpy()
    wind_y = wind_df['wind_y'].to_numpy()

    #### 
    print("Generating indices of plume puff births...")
    births = scipy.stats.poisson.rvs(birth_rate, size=len(T))
    birth_indices = []
    for idx, birth in enumerate(births):
        birth_indices.extend([idx]*birth)

    #### 
    print("Starting parallel simulation ({} cores) for each of {} plume puffs...".format(ncores, len(birth_indices)))
    # Setup inputs for parallel simulation
    seeds = np.arange(0, len(birth_indices))
    puff_indices = np.arange(0, len(birth_indices))
    # OVERRIDE wind_y_var
    # wind_y_var = np.linalg.norm([wind_x, wind_y])/wind_y_varx
    inputs = [[T,
        tidxs,
        wind_x,
        wind_y, 
        wind_y_var,
        birth_indices[i], 
        puff_indices[i], # could just use i
        seeds[i], # could just use i
        ] for i in range(0, len(birth_indices))]

    t_start = time.time()
    pool = multiprocessing.Pool(ncores)
    # puff_dfs = pool.map(integrate_puff_from_birth_df, inputs)
    puff_dfs = list(tqdm.tqdm(pool.imap(integrate_puff_from_birth_df, inputs)))
    comp_time = time.time() - t_start
    if verbose:
        print('Computation time: ', comp_time)
        print('ncores: ', ncores, ' puffs: ', len(birth_indices), ' steps: ', len(T))
        print('Computation time (C) per core (n) per puff (p) per step (s) = (C*n/p*s) = ', ncores*comp_time/(len(birth_indices)*len(T)))
        print('Total time = C*n/p*s')

    t_start = time.time()
    puffs_df = pd.concat(puff_dfs)
    comp_time = time.time() - t_start
    print('Time to concatenate {} puff dataframes: {}'.format(len(puff_dfs), comp_time))

    n_times_pre = len(puffs_df['time'].unique())
    for col in puffs_df.select_dtypes(include='float').columns:
        if 'time' in col:
            puffs_df[col] = puffs_df[col].astype('float64') # time needs a heavier float
        else:
            puffs_df[col] = puffs_df[col].astype('float16') # Use lightest floats
    assert n_times_pre == len(puffs_df['time'].unique()) # Make sure compression lossless

    return puffs_df


#### Faster vectorized version ####
def gen_puff_dict(puff_number, tidx):
    return {
     'puff_number': puff_number,
     'time': np.float64(tidx)/100.,
     'x': 0.0,
     'y': 0.0,
     'radius': 0.01,
     # 'x_minus_radius': -0.01,
     # 'x_plus_radius': 0.01,
     # 'y_minus_radius': -0.01,
     # 'y_plus_radius': 0.01,
     # 'concentration': 1.0,
     'tidx': tidx,
    }
    
def grow_puffs(birth_rate, puff_t, tidx):
    num_births = sp.stats.poisson.rvs(birth_rate, size=1)[0]
    puff_number = puff_t['puff_number'].max() + 1
    
    new_rows = [ gen_puff_dict(puff_number+i, tidx) for i in range(num_births)]    
    new_rows = pd.DataFrame( new_rows )
    return pd.concat([puff_t, new_rows])
    
def manual_integrator(puff_t, wind_t, tidx,
                      dt=np.float64(0.01), 
                      rdot=0.01, 
                      birth_rate=1.0, 
                      min_radius=0.01, 
                      wind_y_var=0.5):
    n_puffs = len(puff_t)
    puff_t['x'] += wind_t['wind_x'].item()*dt
    puff_t['y'] += wind_t['wind_y'].item()*dt + np.random.normal(0, wind_y_var, size=n_puffs)*dt
    puff_t['radius'] += dt * rdot
    puff_t['tidx'] = tidx
    puff_t['time'] = wind_t['time'].item()
    
    # Trim plume
    puff_t = puff_t.query("(radius > 0) & (x<10) & (y<10) & (x>-2) & (y>-10)")

    # Grow plume
    puff_t = grow_puffs(birth_rate, puff_t, tidx)
    
    return puff_t

def get_puffs_df_vector(wind_df, wind_y_var, birth_rate, verbose=True):
    """Fast vectorized euler stepper"""
    print(wind_df.shape, wind_y_var, birth_rate)
    # Initialize
    n_steps = int((wind_df['time'].max() - wind_df['time'].min())*100)
    tidx = 0
    puff_t = pd.DataFrame([gen_puff_dict(puff_number=0, tidx=tidx)])
    wind_t = wind_df.query("tidx == @tidx").copy(deep=True).reset_index(drop=True)

    # Main Euler integrator loop
    puff_dfs = []
    for i in tqdm.tqdm(range(n_steps)):
        puff_t = manual_integrator(puff_t, wind_t, tidx, 
            birth_rate=birth_rate, wind_y_var=wind_y_var)
        puff_dfs.append( puff_t )

        tidx += 1
        wind_t = wind_df.query("tidx == @tidx").copy(deep=True).reset_index(drop=True)
        if wind_t.shape[0] is not 1:
            print("Likely numerical error!:", tidx, wind_t)

    # Gather data and post-process float format
    t_start = time.time()
    puffs_df = pd.concat(puff_dfs)
    if verbose:
        comp_time = time.time() - t_start
        print('Time to concatenate {} puff dataframes: {}'.format(len(puff_dfs), comp_time))

    n_times_pre = len(puffs_df['time'].unique())
    for col in puffs_df.select_dtypes(include='float').columns:
        if 'time' in col:
            puffs_df[col] = puffs_df[col].astype('float64') # time needs a heavier float
        else:
            puffs_df[col] = puffs_df[col].astype('float16') # Use lightest floats
    assert n_times_pre == len(puffs_df['time'].unique()) # Make sure compression lossless
    return puffs_df

