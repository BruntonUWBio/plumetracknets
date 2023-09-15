"""

python centerline_cli.py --dataset constantx5b5

for DATASET in constantx5b5 switch45x5b5 noisy3x5b5; do
  python3 centerline_cli.py --dataset $DATASET &
done
"""


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import pandas as pd
import config
import numpy as np
import tqdm

def gen_puff_dict_centerline(puff_number, tidx):
    return {
     'puff_number': puff_number,
     'time': np.float64(tidx)/100.,
     'x': 0.0,
     'y': 0.0,
     'tidx': tidx,
    }
    
def grow_puffs_centerline(puff_t, tidx):
    num_births = 1
    puff_number = puff_t['puff_number'].max() + 1
    
    new_rows = [ gen_puff_dict_centerline(puff_number+i, tidx) for i in range(num_births)]    
    new_rows = pd.DataFrame( new_rows )
    return pd.concat([puff_t, new_rows])
    
def manual_integrator_centerline(puff_t, wind_t, tidx,
                      dt=np.float64(0.01)):
    n_puffs = len(puff_t)
    puff_t['x'] += wind_t['wind_x'].item()*dt
    puff_t['y'] += wind_t['wind_y'].item()*dt
    puff_t['tidx'] = tidx
    puff_t['time'] = wind_t['time'].item()

    # Trim plume
    puff_t = puff_t.query("(x<10) & (y<10) & (x>-2) & (y>-10)")

    # Grow plume
    puff_t = grow_puffs_centerline(puff_t, tidx)    

    return puff_t

def get_puffs_df_vector_centerline(wind_df, verbose=True):
    """Fast vectorized euler stepper"""
    # Initialize
    n_steps = int((wind_df['time'].max() - wind_df['time'].min())*100)
    tidx = 0
    puff_t = pd.DataFrame([gen_puff_dict_centerline(puff_number=0, tidx=tidx)])
    wind_t = wind_df.query("tidx == @tidx").copy(deep=True).reset_index(drop=True)

    # Main Euler integrator loop
    puff_dfs = []
    for i in tqdm.tqdm(range(n_steps)):
        puff_t = manual_integrator_centerline(puff_t, wind_t, tidx)
        puff_dfs.append( puff_t )
        tidx += 1
        wind_t = wind_df.query("tidx == @tidx").copy(deep=True).reset_index(drop=True)
        if wind_t.shape[0] != 1:
            print("Likely numerical error!:", tidx, wind_t)

    # Gather data and post-process float format
    puffs_df = pd.concat(puff_dfs)
    n_times_pre = len(puffs_df['time'].unique())
    for col in puffs_df.select_dtypes(include='float').columns:
        if 'time' in col:
            puffs_df[col] = puffs_df[col].astype('float64') # time needs a heavier float
        else:
            puffs_df[col] = puffs_df[col].astype('float16') # Use lightest floats
    assert n_times_pre == len(puffs_df['time'].unique()) # Make sure compression lossless
    return puffs_df



# def augment_slopes(centerline_df):
#     gb = centerline_df.groupby(by='puff_number')
#     # bypuffdiff = bypuff.apply(np.diff)

#     gbp_list = [gb.get_group(x) for x in tqdm.tqdm(gb.groups)]
#     gbp_list[0]

#     # centerline_df['slope'] = centerline_df['y'].diff()/centerline_df['x'].diff()
#     for gbp in tqdm.tqdm(gbp_list):
#     #     gbp['slope'] = gbp['y'].diff()/gbp['x'].diff()
#         gbp['slope'] = gbp['y'].rolling(8).mean().diff()/gbp['x'].rolling(8).mean().diff()
#         gbp.fillna(0, inplace=True)
#         gbp['slope'] = np.arctan(gbp['slope'])

#     gbp_list[0]
    
#     centerline_df = pd.concat(gbp_list).sort_values(by=['tidx', 'puff_number']).reset_index(drop=True)
#     return centerline_df

# centerline_df = augment_slopes(centerline_df)


if __name__ == '__main__':
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description='Gen centerline file')
    parser.add_argument('--dataset',  type=str, default='test')
    args = parser.parse_args()
    print(args)

    dataset = args.dataset

    data_dir=config.datadir
    wind_filename = f'{data_dir}/wind_data_{dataset}.pickle' 
    wind_df = pd.read_pickle(wind_filename)

    centerline_df = get_puffs_df_vector_centerline(wind_df, verbose=True)

    centerline_df = centerline_df.sort_values(by=['tidx', 'puff_number']).reset_index(drop=True)

    # centerline_df['slope'] = centerline_df['y'].diff()/centerline_df['x'].diff()
    y_diff = centerline_df['y'].rolling(8).mean().diff()
    x_diff = centerline_df['x'].rolling(8).mean().diff()
    centerline_df['slope'] = y_diff/x_diff
    # centerline_df['angle'] = np.arctan2(y_diff, x_diff)/np.pi
    centerline_df['angle'] = np.arctan(centerline_df['slope'])/np.pi
    centerline_df['angle'] = (centerline_df['angle'] + 1)/2 # shift scale (log_analysis)
    # centerline_df['angle'] = log_analysis.shift_scale_theta(log_analysis.wind_xy_to_theta(x_diff, y_diff)) 


    centerline_filename = f'{data_dir}/centerline_data_{dataset}.pickle' 
    centerline_df.to_pickle(centerline_filename)