"""
python -u sim_cli.py --duration 10 # test ~20s
python -u sim_cli.py --duration 120 # test ~3.5min

# Real datasets, T=120s is enough!
python -u sim_cli.py --duration 120 --dataset_name constant 
python -u sim_cli.py --duration 120 --dataset_name noisy3

python -u sim_cli.py --duration 120 --dataset_name constant --fname_suffix x2 --wind_magnitude 0.2
python -u sim_cli.py --duration 120 --dataset_name noisy3 --fname_suffix x2 --wind_magnitude 0.2

# Wind magnitude X
for X in 1 2 3 4 5; do
  python -u sim_cli.py --duration 120 --dataset_name constant --fname_suffix x${X} --wind_magnitude 0.${X}
  python -u sim_cli.py --duration 120 --dataset_name noisy3 --fname_suffix x${X} --wind_magnitude 0.${X}
done

# Higher birthrates @ speed 5x
# for BX in 1 2 3 4 5; do
for BX in 5; do
 for DATASET in noisy1 noisy2 noisy3 noisy4 noisy5 noisy6; do
  BR=$(echo "0.2*${BX}" | bc)
  echo $DATASET $BR 
  python -u sim_cli.py \
    --duration 120 \
    --dataset_name $DATASET \
    --wind_magnitude 0.5 \
    --birth_rate ${BR} \
    --fname_suffix x5b${BX} > ${DATASET}x5b${BX}.log 2>&1 &
 done
done

# Test
python sim_cli.py --duration 10 --dataset_name noisy1 \
    --wind_magnitude 0.5 --birth_rate 0.2 --fname_suffix x5b1

# Plume width/narrowness -- higher the YVARX, the narrower the plume
BX=5
for YVARX in 1 4 9 16 64; do 
  BR=$(echo "0.2*${BX}" | bc)
  echo BR: $BR YVARX: $YVARX

  python -u sim_cli.py --duration 120 --dataset_name constant --wind_y_varx $(echo "1*${YVARX}" | bc) --fname_suffix x5b${BX}v${YVARX} --wind_magnitude 0.5 --birth_rate ${BR} > constantx5b${BX}v${YVARX}.log 2>&1 &
done


## test new euler-step simulator ##
python -u sim_cli.py --duration 120 --dataset_name constant --fname_suffix x5b5new --wind_magnitude 0.5 --birth_rate 1.0

python -u sim_cli.py --duration 120 --dataset_name noisy6 --fname_suffix x5b5limited --wind_magnitude 0.5 --birth_rate 1.0


# Higher birthrates @ speed 5x
# for BX in 1 2 3 4 5; do
for BX in 5; do
 for DATASET in const2noisy1 const2noisy2 const2noisy3 const2noisy4 const2noisy5 const2noisy6; do
  BR=$(echo "0.2*${BX}" | bc)
  echo $DATASET $BR 
  python -u sim_cli.py \
    --duration 120 \
    --dataset_name $DATASET \
    --wind_magnitude 0.5 \
    --birth_rate ${BR} \
    --fname_suffix x5b${BX} > ${DATASET}x5b${BX}.log 2>&1 &
 done
done



"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sim_utils
import argparse
import pandas as pd
import config
import numpy as np

# Parse CLI arguments
parser = argparse.ArgumentParser(description='Generate plume simulations')
parser.add_argument('--duration',  metavar='d',  type=int, 
  help='simulation duration in seconds', default=15)
parser.add_argument('--cores',  metavar='c',  type=int, 
  help='number of cores to use', default=24)
parser.add_argument('--dataset_name',  type=str, default='test')
parser.add_argument('--fname_suffix',  type=str, default='')
parser.add_argument('--dt',  type=float, 
	help='time per step (seconds)', default=0.01)
parser.add_argument('--wind_magnitude',  type=float, 
	help='m/s', default=0.1)
parser.add_argument('--wind_y_varx',  type=float, default=1.0)
parser.add_argument('--birth_rate',  type=float, 
	help='poisson birth_rate parameter', default=0.2)
parser.add_argument('--outdir',  type=str, default=config.datadir)

args = parser.parse_args()
print(args)

wind_df = sim_utils.get_wind_xyt(
	args.duration+1, 
	dt=args.dt,
	wind_magnitude=args.wind_magnitude,
	regime=args.dataset_name
	)
wind_df['tidx'] = np.arange(len(wind_df), dtype=int) 
fname = f'{args.outdir}/wind_data_{args.dataset_name}{args.fname_suffix}.pickle'
wind_df.to_pickle(fname)
print(wind_df.head(n=5))
print(wind_df.tail(n=5))
print("Saved", fname)


# Older ODEINT version
# wind_y_var = args.wind_magnitude/np.sqrt(args.wind_y_varx)
# puff_df = sim_utils.get_puffs_df_oneshot(wind_df, wind_y_var,
# 	args.birth_rate, args.cores, verbose=True)

# Using faster vectorized version
wind_y_var = args.wind_magnitude/np.sqrt(args.wind_y_varx)
puff_df = sim_utils.get_puffs_df_vector(wind_df, wind_y_var, args.birth_rate, verbose=True)

fname = f'{args.outdir}/puff_data_{args.dataset_name}{args.fname_suffix}.pickle'
puff_df.to_pickle(fname)
print('puff_df.shape', puff_df.shape)
print(puff_df.tail())
print(puff_df.head())
print("Saved", fname)


## -- Extra Viz -- ##
# Plot puffs - also serves a good test
# Need to add concentration & radius data before plotting
import sim_analysis # load config later, eek!
data_puffs, data_wind = sim_analysis.load_plume(f'{args.dataset_name}{args.fname_suffix}')
t_val = data_puffs['time'].iloc[-1]
fig, ax = sim_analysis.plot_puffs_and_wind_vectors(
	data_puffs, 
	data_wind, 
	t_val, 
    fname='', 
    plotsize=(8,8))
fig.savefig(f'{args.outdir}/{args.dataset_name}{args.fname_suffix}_t{t_val:3.3f}.png')
ax.set_xlim(-1, 12)
ax.set_ylim(-1.8, +1.8)
if 'switch' in args.dataset_name:
    ax.set_xlim(-1, +10) # if switching
    ax.set_ylim(-5, +5) # if switching
fig.savefig(f'{args.outdir}/{args.dataset_name}{args.fname_suffix}_t{t_val:3.3f}z.png')

