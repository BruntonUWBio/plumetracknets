import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
import scipy.interpolate
import pandas
import pandas as pd
import time
import multiprocessing
import os
import matplotlib 
import config
np.random.seed(config.seed_global)

import mpl_scatter_density

def get_puff_birthtime(data_puffs, puff_number):
    p = data_puffs[data_puffs.puff_number==puff_number].time.values
    p = p[np.where(p>0)]
    return np.min(p)

def make_odor_array(inputs):
    data_puffs, tval, xmin, xmax, ymin, ymax, meshsize = inputs
    data_at_t = data_puffs[data_puffs.time == tval]
    x = np.linspace(xmin, xmax, 300)
    y = np.linspace(ymin, ymax, 300)
    xv, yv = np.meshgrid(x, y)
    odor_array = np.zeros_like(xv)

    for puff_number in data_at_t.puff_number.unique():
        puff = data_at_t[data_at_t.puff_number == puff_number]
        xidx = np.where( (xv>puff.x_minus_radius.values)*(xv<puff.x_plus_radius.values))[1]
        yidx = np.where( (yv>puff.y_minus_radius.values)*(yv<puff.y_plus_radius.values))[0]
        if len(xidx) > 0 and len(yidx) > 0:
            odor_array[xidx[0]:xidx[-1], yidx[0]:yidx[-1]] += puff.concentration.values

    with open('odor_arrays/odor_array_'+str(int(tval*100)).zfill(5)+'.npy', 'wb') as f:
        np.save(f, odor_array)

def parallel_make_odor_arrays(data_puffs, T, meshsize=300):
    xmin = data_puffs.x.min()
    xmax = data_puffs.x.max()
    ymin = data_puffs.y.min()
    ymax = data_puffs.y.max()

    stride = 26
    for i in range(0, len(T), stride):
        t_start = time.time()
        inputs = [[ copy.copy(data_puffs[data_puffs.time == tval]), tval, xmin, xmax, ymin, ymax, meshsize] for tval in T[i:i+stride]]
        ncores = stride
        pool = multiprocessing.Pool(ncores)
        puffs = pool.map(make_odor_array, inputs)
        comp_time = time.time() - t_start
        pool.terminate()
        print('i: ', i, ' Computation time: ', comp_time)


def make_odor_age_array(inputs):
    data_puffs, tval, xmin, xmax, ymin, ymax, meshsize = inputs
    data_at_t = data_puffs[data_puffs.time == tval]
    x = np.linspace(xmin, xmax, 300)
    y = np.linspace(ymin, ymax, 300)
    xv, yv = np.meshgrid(x, y)
    odor_array = np.zeros_like(xv)

    for puff_number in data_at_t.puff_number.unique():
        puff = data_at_t[data_at_t.puff_number == puff_number]
        xidx = np.where( (xv>puff.x_minus_radius.values)*(xv<puff.x_plus_radius.values))[1]
        yidx = np.where( (yv>puff.y_minus_radius.values)*(yv<puff.y_plus_radius.values))[0]
        if len(xidx) > 0 and len(yidx) > 0:
            odor_array[xidx[0]:xidx[-1], yidx[0]:yidx[-1]] = tval-get_puff_birthtime(data_puffs, puff_number)

    with open('odor_ages/odor_ages_'+str(int(tval*100)).zfill(5)+'.npy', 'wb') as f:
        np.save(f, odor_array)


def parallel_make_odor_age_arrays(data_puffs, T, meshsize=300):
    xmin = data_puffs.x.min()
    xmax = data_puffs.x.max()
    ymin = data_puffs.y.min()
    ymax = data_puffs.y.max()

    stride = 26
    for i in range(0, len(T), stride):
        t_start = time.time()
        inputs = [[ copy.copy(data_puffs[data_puffs.time == tval]), tval, xmin, xmax, ymin, ymax, meshsize] for tval in T[i:i+stride]]
        ncores = stride
        pool = multiprocessing.Pool(ncores)
        puffs = pool.map(make_odor_age_array, inputs)
        comp_time = time.time() - t_start
        pool.terminate()
        print('i: ', i, ' Computation time: ', comp_time)


#######################################################################
### Helper functions

def diffa(array):
    d = np.diff(array)
    d = np.hstack( (d[0], d) )
    return d

def get_continuous_chunks(array, array2=None, jump=1, return_index=False):
    """
    Splits array into a list of continuous chunks. Eg. [1,2,3,4,5,7,8,9] becomes [[1,2,3,4,5], [7,8,9]]
    
    array2  -- optional second array to split in the same way array is split
    jump    -- specifies size of jump in data to create a break point
    """
    diffarray = diffa(array)
    break_points = np.where(np.abs(diffarray) > jump)[0]
    break_points = np.insert(break_points, 0, 0)
    break_points = np.insert(break_points, len(break_points), len(array))
    
    chunks = []
    array2_chunks = []
    index = []
    for i, break_point in enumerate(break_points):
        if break_point >= len(array):
            break
        chunk = array[break_point:break_points[i+1]]
        if type(chunk) is not list:
            chunk = chunk.tolist()
        chunks.append(chunk)
        
        if array2 is not None:
            array2_chunk = array2[break_point:break_points[i+1]]
            if type(array2_chunk) is not list:
                array2_chunk = array2_chunk.tolist()
            array2_chunks.append(array2_chunk)
        
        if return_index:
            indices_for_chunk = np.arange(break_point,break_points[i+1])
            index.append(indices_for_chunk)
            
    if type(break_points) is not list:
        break_points = break_points.tolist()
        
    if return_index:
        return index
    
    if array2 is None:
        return chunks, break_points
    
    else:
        return chunks, array2_chunks, break_points
    
#######################################################################
### Analysis




def load_plume(
    dataset='constant', 
    t_val_min=None,
    t_val_max=None,
    env_dt=0.04,
    puff_sparsity=1.00,
    radius_multiplier=1.00,
    diffusion_multiplier=1.00,
    data_dir=config.datadir,
    ):
    print("[load_plume]",dataset)
    puff_filename = f'{data_dir}/puff_data_{dataset}.pickle' 
    wind_filename = f'{data_dir}/wind_data_{dataset}.pickle' 

    # pandas dataframe
    data_puffs = pandas.read_pickle(puff_filename)
    data_wind = pandas.read_pickle(wind_filename)

    # Load plume/wind data and truncate away upto t_val_min 
    if t_val_min is not None:
        data_wind.query("time >= {}".format(t_val_min), inplace=True)
        data_puffs.query("time >= {}".format(t_val_min), inplace=True)

    # SPEEDUP: **Further** truncate plume/wind data by sim. time
    if t_val_max is not None:
        data_wind.query("time <= {}".format(t_val_max), inplace=True)
        data_puffs.query("time <= {}".format(t_val_max), inplace=True)

    ## Downsample to env_dt!
    env_dt_int = int(env_dt*100)
    assert env_dt_int in [2, 4, 5, 10] # Limit downsampling to these for now!
    if 'tidx' not in data_wind.columns:
    	data_wind['tidx'] = (data_wind['time']*100).astype(int)
    if 'tidx' not in data_puffs.columns:
    	data_puffs['tidx'] = (data_puffs['time']*100).astype(int)
    data_wind.query("tidx % @env_dt_int == 0", inplace=True)
    data_puffs.query("tidx % @env_dt_int == 0", inplace=True)

    # Sparsify puff data (No change in wind)
    if puff_sparsity < 0.99:
        print(f"[load_plume] Sparsifying puffs to {puff_sparsity}x")
        puff_sparsity = np.clip(puff_sparsity, 0.0, 1.0)
        drop_idxs = data_puffs['puff_number'].unique()
        drop_idxs = pd.Series(drop_idxs).sample(frac=(1.00-puff_sparsity))
        data_puffs.query("puff_number not in @drop_idxs", inplace=True)

    # Multiply radius 
    if radius_multiplier != 1.0:
        print("Applying radius_multiplier", radius_multiplier)
        data_puffs.loc[:,'radius'] *= radius_multiplier

    min_radius = 0.01

    # Adjust diffusion rate
    if diffusion_multiplier != 1.0:
        print("Applying diffusion_multiplier", diffusion_multiplier)
        data_puffs.loc[:,'radius'] -= min_radius # subtract initial radius
        data_puffs.loc[:,'radius'] *= diffusion_multiplier # adjust 
        data_puffs.loc[:,'radius'] += min_radius # add back initial radius

    # Add other columns
    data_puffs['x_minus_radius'] = data_puffs.x - data_puffs.radius
    data_puffs['x_plus_radius'] = data_puffs.x + data_puffs.radius
    data_puffs['y_minus_radius'] = data_puffs.y - data_puffs.radius
    data_puffs['y_plus_radius'] = data_puffs.y + data_puffs.radius
    data_puffs['concentration'] = (min_radius/data_puffs.radius)**3


    return data_puffs, data_wind

def calculate_concentrations(data):
    # print("Using simpler calculate_concentrations()....")
    rad = data.radius
    min_radius = min(rad)
    assert min_radius >= 0.01
    # c = 1/(4/3.*np.pi*(rad.astype('float32'))**3)
    c = (min_radius/rad)**3
    data.loc[:,'concentration'] = c
    return data

def calculate_concentrations_floris(data):
    rad = data.radius.values
    rad[rad==0] = np.inf
    min_radius = np.min(rad)
    c = 1/(4/3.*np.pi*rad**3)
    c[np.isinf(c)] = 0
    c /= (1/(4/3*np.pi*min_radius**3))
    data['concentration'] = c
    return data

def get_concentration_at_point_in_time_pandas(data, t_val, x_val, y_val):
    # find the indices for all puffs that intersect the given x,y,time point
    qx = str(x_val) + ' > x_minus_radius and ' + str(x_val) + ' < x_plus_radius'
    qy = str(y_val) + ' > y_minus_radius and ' + str(y_val) + ' < y_plus_radius'
    q = qx + ' and ' + qy
    d = data[data.time==t_val].query(q)
    # d = data[np.isclose(data.time, t_val, atol=1e-3)] # BROKEN! Smallest dt=0.01, so this is more than enough!

    return d.concentration.sum()

def get_concentration_at_tidx(data, tidx, x_val, y_val):
    # find the indices for all puffs that intersect the given x,y,time point
    qx = str(x_val) + ' > x_minus_radius and ' + str(x_val) + ' < x_plus_radius'
    qy = str(y_val) + ' > y_minus_radius and ' + str(y_val) + ' < y_plus_radius'
    q = qx + ' and ' + qy
    d = data[data.tidx==tidx].query(q)
    return d.concentration.sum()


def get_concentration_at_point_in_time_arrays(directory, t_val, x_val, y_val, xgrid=None, ygrid=None, data_puffs=None):
    # xval and yval should be a pair of start and end points of the trajectory

    if xgrid is None or ygrid is None:
        assert data_puffs is not None
        xmin = int(data_puffs.x.min())
        xmax = int(data_puffs.x.max())
        ymin = int(data_puffs.y.min())
        ymax = int(data_puffs.y.max())
        x = np.linspace(xmin, xmax, 300)
        y = np.linspace(ymin, ymax, 300)
        xgrid, ygrid = np.meshgrid(x, y)

    basename = os.path.dirname(directory)[0:-1]
    if 'age' in basename:
        basename += 's'
    basename += '_'

    filename = os.path.join(directory, basename + str(int(100*t_val)).zfill(5) + '.npy')
    odor_array = np.load(filename).T

    x = np.linspace(x_val[0], x_val[-1], 10)
    y = np.linspace(y_val[0], y_val[-1], 10)

    idx = []
    for i in range(len(x)):
        idx.append(np.where( (np.abs(xgrid-x[i])<0.1)*(np.abs(ygrid-y[i])<0.1) ))

    cs = []
    for ix in idx:
        cs.extend(odor_array[ix].tolist())
    c = np.mean(cs)

    return c 


def get_random_trajectory(n_pts=20, xmin=-1, xmax=2.5, ymin=-1, ymax=2.5, tstart=0, tend=100, dt=0.01, local_state=None):
    if local_state is None:
        local_state = np.random.RandomState(np.random.randint(0,1000))
    N = n_pts
    t = np.linspace(tstart,tend,N)
    x = local_state.uniform(xmin, xmax, N)
    y = local_state.uniform(ymin, ymax, N)
    
    spl = scipy.interpolate.splrep(t, x)
    t2 = np.arange(tstart,tend,dt)
    x2 = scipy.interpolate.splev(t2, spl)

    spl = scipy.interpolate.splrep(t, y)
    t2 = np.arange(tstart,tend,dt)
    y2 = scipy.interpolate.splev(t2, spl)
    
    return x2, y2, t2

def get_odor_values_along_trajectory(data, x, y):
    odor = []
    for i in range(len(x)):
        o = get_concentration_at_point_in_time(data, i, x[i], y[i])
        odor.append(o)
    return np.array(odor)

def get_whiff_length_and_interval_for_trajectory(x, y, odor, threshold=0.001):
    '''
    For each whiff, return the (x,y) location of the middle of the whiff, length, mean time elapsed since last and before next, mean concentration, peak concentration
    '''

    whiffs_continuous = np.where(odor>threshold)[0]
    __whiffs_idx__, break_pts = get_continuous_chunks(whiffs_continuous)

    whiffs_idx = []
    for idx in __whiffs_idx__:
        if len(idx) > 2:
            whiffs_idx.append(idx)

    whiff_data = []
    for i, idx in enumerate(whiffs_idx):
        whiff = {'x': None, 'y': None, 'length': None, 'mean_concentration': None, 'max_concentration': None}
        whiff['x'] = x[int(np.nanmean(idx))]
        whiff['y'] = y[int(np.nanmean(idx))]
        v = np.sqrt(np.diff(x[idx])**2 + np.diff(y[idx])**2)
        whiff['length'] = len(idx)*np.mean(v)
        


        whiff['mean_concentration'] = np.nanmean(odor[idx])
        whiff['max_concentration'] = np.nanmax(odor[idx])

        whiff_data.append(whiff)

    return whiff_data


def get_whiff_age(data, t_idx, x, y):
    x_val = x
    y_val = y
    puffs = data['puffs']
    idx = np.where( (np.abs(puffs[:,t_idx,0]-x_val) < np.abs(puffs[:,t_idx,2]))*(np.abs(puffs[:,t_idx,1]-y_val) < np.abs(puffs[:,t_idx,2])) )
    idx = idx[0][np.where(np.invert(np.isinf(puffs[idx,t_idx,2])))[0]]

    t_idx_ages = []

    for i in idx:
        if np.min(puffs[i, :, 0]) > 0: # puff born right away!
            t_idx_birth = 0
        else:
            t_idx_birth = np.where(puffs[i, :, 0]!=0)[0][0]
        t_idx_age = t_idx - t_idx_birth
        t_idx_ages.append(t_idx_age)

    dt = np.mean(np.diff(data['time']))

    return np.mean(t_idx_ages)*dt


#######################################################################
### Plotting

# New version: One circle at hardcoded (x,y) with quiver 
def plot_wind_vectors(data_puffs, data_wind, t_val, ax):
    # Instantaneous wind velocity
    # Normalize wind (just care about angle)
    data_at_t = data_wind[data_wind.time==t_val]
    v_x, v_y = data_at_t.wind_x.mean(), data_at_t.wind_y.mean()
    v_xy = np.sqrt(v_x**2 + v_y**2)*20
    v_x, v_y = v_x/v_xy, v_y/v_xy
    # print("v_x, v_y", v_x, v_y)

    # Arrow
    x,y = -0.15, 0.6 # Arrow Center [Note usu. xlim=(-0.5, 8)]
    ax.quiver(x, y, v_x, v_y, color='black', scale=2.5)
    # ax.quiver(x, y, v_x, v_y, color='black', scale=500)

    # Circle is 1 scatterplot point!
    ax.scatter(x, y, s=500, 
        facecolors='none', 
        edgecolors='k',
        linestyle='--')

# Floris version: fills arena with quivers
# def plot_wind_vectors_floris(data_puffs, data_wind, t_val, ax=None):
#     if ax is None:
#         fig = plt.figure()
#         ax = fig.add_subplot(111)

#     xmin = -5 #data.x.min()
#     xmax = 20 #data.x.max()
#     ymin = -5 #data.y.min()
#     ymax = 20 #data.y.max()

#     data_at_t = data_wind[data_wind.time==t_val]

#     num_grid = 45
#     x = np.linspace(xmin, xmax, num_grid)
#     y = np.linspace(ymin, ymax, num_grid)

#     xv, yv = np.meshgrid(x, y)

#     ax.quiver(x, y, data_at_t.wind_x.mean()**np.ones([len(x), len(y)]), 
#                     data_at_t.wind_y.mean()**np.ones([len(x), len(y)]),
#                     color='black', scale=0.5)


def plot_puffs(data, t_val, ax=None, show=True):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
        
    # xmin = -2 #data.x.min()
    # xmax = 12 #data.x.max()
    # ymin = -5 #data.y.min()
    # ymax = +5 #data.y.max()
    # set limits
    # ax.set_xlim(xmin, xmax)
    # ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')

    # data_at_t = data[data.time==t_val] # Float equals is dangerous!
    data_at_t = data[np.isclose(data.time, t_val, atol=1e-3)] # Smallest dt=0.01, so this is more than enough!
    # print("data_at_t.shape", data_at_t.shape, t_val, data.time.min(), data.time.max())

    c = data_at_t.concentration
    # print(c, t_val)

    # alphas = (np.log(c+1e-5)+np.abs(np.log(1e-5))).values
    # alphas /= np.max(alphas)
    # alphas = np.clip(alphas, 0.0, 1.0)

    alphas = c.values
    alphas /= np.max(alphas) # 0...1
    alphas = np.power(alphas, 1/8) # See minimal2 notebook
    # alphas = np.power(alphas, 10)
    alphas = np.clip(alphas, 0.2, 0.4)

    alphas *= 2.5/data_at_t.x # decay alpha by distance too
    alphas = np.clip(alphas, 0.05, 0.4)


    rgba_colors = np.zeros((data_at_t.time.shape[0],4))
    # rgba_colors[:,0] = 1.0 # Red
    # rgba_colors[:,2] = 1.0 # Blue
    # https://matplotlib.org/3.1.1/gallery/color/named_colors.html
    # https://matplotlib.org/3.1.0/tutorials/colors/colors.html
    rgba_colors[:,0:3] = matplotlib.colors.to_rgba('gray')[:3] # decent
    # rgba_colors[:,0:3] = matplotlib.colors.to_rgba('darkgray')[:3] # decent
    # rgba_colors[:,0:3] = matplotlib.colors.to_rgba('dimgray')[:3] # decent
    # rgba_colors[:,0:3] = matplotlib.colors.to_rgba('darkslategray')[:3] # too dark
    # rgba_colors[:,0:3] = matplotlib.colors.to_rgba('lightsteelblue')[:3] # ok
    # rgba_colors[:,0:3] = matplotlib.colors.to_rgba('red')[:3] 
    # rgba_colors[:,0:3] = matplotlib.colors.to_rgba('lightskyblue')[:3] 

    # the fourth column needs to be your alphas
    rgba_colors[:, 3] = alphas

    # fig.canvas.draw()
    # s = ((ax.get_window_extent().width  / (xmax-xmin+1.) * 72./fig.dpi) ** 2)
    k = 6250*((fig.get_figwidth()/8.0)**2) # trial-and-error
    s = k*(data_at_t.radius)**2 
    # print('size', s) # 885

    ax.scatter(data_at_t.x, data_at_t.y, s=s, facecolor=rgba_colors, edgecolor='none')

    if show:
        plt.show()

# Floris' version 
# def plot_puffs_floris(data, t_val, ax=None, show=True):
#     if ax is None:
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#     else:
#         fig = ax.figure
        
#     xmin = -2 #data.x.min()
#     xmax = 12 #data.x.max()
#     ymin = -5 #data.y.min()
#     ymax = +5 #data.y.max()
    # data_at_t = data[np.isclose(data.time, t_val, atol=1e-3)] # Smallest dt=0.01, so this is more than enough!
#     data_at_t = data[data.time==t_val]

#     c = data_at_t.concentration
#     # print(c, t_val)

#     alphas = (np.log(c+1e-5)+np.abs(np.log(1e-5))).values
#     alphas /= np.max(alphas)
#     alphas[np.where(alphas<0)] = 0
#     alphas[np.where(alphas>1)] = 1    
#     min_alpha = 0.3 # ALIFE
#     alphas[np.where(alphas<min_alpha)] = min_alpha # ALIFE

#     rgba_colors = np.zeros((data_at_t.time.shape[0],4))
#     # rgba_colors[:,0] = 1.0 # Red
#     # rgba_colors[:,2] = 1.0 # Blue
#     # https://www.rapidtables.com/web/color/gray-color.html
#     # rgba_colors[:,0:3] = matplotlib.colors.to_rgba('red')[:2] 
#     rgba_colors[:,0:3] = matplotlib.colors.to_rgba('gray')[:2] 

#     # the fourth column needs to be your alphas
#     rgba_colors[:, 3] = alphas

#     # set limits
#     ax.set_xlim(xmin, xmax)
#     ax.set_ylim(ymin, ymax)
#     ax.set_aspect('equal')

#     fig.canvas.draw()
#     s = ((ax.get_window_extent().width  / (xmax-xmin+1.) * 72./fig.dpi) ** 2)

#     ax.scatter(data_at_t.x, data_at_t.y, s=s*(data_at_t.radius)**2, facecolor=rgba_colors, edgecolor='none')

#     if show:
#         plt.show()


def plot_puffs_and_wind_vectors(data_puffs, data_wind, t_val, ax=None, fname='', plotsize=(10,10), show=True):
    if ax is None:
        fig = plt.figure(figsize=plotsize)
        ax = fig.add_subplot(111)
    
    plot_wind_vectors(data_puffs, data_wind, t_val, ax)
    plot_puffs(data_puffs, t_val, ax, show=False)
    
    if len(fname) > 0:
        # fname = savedir + '/' + 'puff_animation_' + str(idx).zfill(int(np.log10(data['puffs'].shape[1]))+1) + '.jpg'
        fig.savefig(fname, format='jpg', bbox_inches='tight')
        plt.close()
    return fig, ax

def make_animation(data):

    for idx in range(data['puffs'].shape[1]):
        plot_puffs_and_wind_vectors(data, idx, savedir='plume_animation')
        
    print( "ffmpeg -n -i '"'animation_%04d.jpg'"' animation.m4v")
    print( "might need: https://www.faqforge.com/linux/how-to-install-ffmpeg-on-ubuntu-14-04/")

def plot_interpolated_grid(x, y, val, resolution=50, xmin=-0.5, xmax=2.5, ymin=-0.5, ymax=2.5):
    xcoords = np.linspace(xmin, xmax, int(resolution/10))
    ycoords = np.linspace(ymin, ymax, int(resolution/10))

    # vertical edges
    if 0:
        for xm in xcoords:
            x = np.hstack(([xm]*len(ycoords), x))
            y = np.hstack((ycoords, y))
            val = np.hstack(([np.min(val)]*len(ycoords), val))
        # horizontal edges
        for ym in ycoords:
            y = np.hstack(([ym]*len(xcoords), y))
            x = np.hstack((xcoords, x))
            val = np.hstack(([np.min(val)]*len(xcoords), val))

    xcoords = np.linspace(xmin, xmax, resolution)
    ycoords = np.linspace(ymin, ymax, resolution)

    X, Y = np.meshgrid(xcoords, ycoords)
    grid = scipy.interpolate.griddata(np.array(np.vstack((x, y))).T, np.array(val), (X, Y), method='linear', fill_value=np.min(val))
    plt.imshow(grid, origin='lower')

def plot_probability_of_whiff(df_whiff, df_data, xmin=-0.5, xmax=2.5, ymin=-0.5, ymax=2.5):
    r_encounters = np.histogram2d(df_whiff.x, df_whiff.y, bins=100, normed=False)
    r_passthrough = np.histogram2d(df_data.x.values, df_data.y.values, bins=100, normed=False)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.imshow(np.log(r_encounters[0]/r_passthrough[0]+1e-3), origin='lower')
    
    #ax.set_xlim(xmin, xmax)
    #ax.set_ylim(ymin, ymax)

    ax.set_aspect('equal')

def plot_mean_concentration(df_data, xmin=-0.5, xmax=2.5, ymin=-0.5, ymax=2.5, resolution=50):
    '''
    This is super slow.. there must be a better way.. I tried scipy.interpolate.griddata and weighted histograms
    Both were incorrect, they do not seem to interpolate duplicate data points correctly
    '''

    def interp(df_data, x, y, eps):
        query_x = 'x < '+str(x+eps)+' and x > '+str(x-eps)
        query_y = 'y < '+str(y+eps)+' and y > '+str(y-eps)
        return df_data.query(query_x+' and ' + query_y).concentration.mean()
        
    xcoords = np.linspace(xmin, xmax, resolution)
    ycoords = np.linspace(ymin, ymax, resolution)
    eps = np.mean(np.diff(xcoords))
    
    concentration = np.zeros([len(xcoords), len(ycoords)])
    for i, x in enumerate(xcoords):
        for j, y in enumerate(ycoords):
            concentration[i,j] = interp(df_data, x, y, eps)
        
        
    fig = plt.figure()
    ax = fig.add_subplot(111)

    v = np.log(concentration)
    v[np.isinf(v)] = np.min(v[~np.isinf(v)])
    v[np.isnan(v)] = np.min(v)


    ax.imshow( v, origin='lower')

    ax.set_aspect('equal')


#### Centerline get/regenerate ####




if __name__ == '__main__':
    data_puffs, data_wind = load_plume(
        'puff_data_switching_wind.pickle', 
        'wind_data_switching_wind.pickle')
    T = data_wind.time.values
    parallel_make_odor_arrays(data_puffs, T, meshsize=300)