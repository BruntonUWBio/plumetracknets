# python3 tabulate --base_dir /src/TrainigCurriculum/data/TrainingCurriculum/sw_dist/

# aggregate performance eval results of all models in a directory 

import tqdm
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import glob
import datetime

def parse_summary_files(fnames, BASE_DIR):
    counts_df = []
    for fname in tqdm.tqdm(fnames):
        s = pd.read_csv(fname) 
        dataset = str(fname).split('/')[-1].replace('_summary.csv','')
        row = {
            'dataset': dataset,
            'HOME': sum(s['reason'] == 'HOME'),
            'OOB': sum(s['reason'] == 'OOB'),
            'OOT': sum(s['reason'] == 'OOT'),
            'total': len(s['reason']),
            'seed': str(fname).split('seed')[-1].split('/')[0],
            'model_dir': str(fname).replace(f'{dataset}_summary.csv','').replace(os.path.expanduser(BASE_DIR), ''),
            'code': str(fname).split('code')[-1].split('_')[0],
            'fname': str(fname)
        }
        counts_df.append(row)
        
    counts_df = pd.DataFrame(counts_df)
    counts_df['relative_plume_density'] = [ds.split("_")[1] if len(ds.split("_")) == 2 else 1 for ds in counts_df['dataset']]
    counts_df['condition'] = [ds.split("_")[0] for ds in counts_df['dataset']]
    counts_df['Success_pct'] = counts_df['HOME'] / counts_df['total'] * 100
    # pivot_df = counts_df.pivot(index=['model_dir', 'seed'], columns='dataset', values='HOME').reset_index()
    return counts_df

def main(args):
    if args.subdir_prefix:
        files = glob.glob(f"{args.base_dir}/**/{args.subdir_prefix}*/**/*_summary.csv", recursive=True)
        # files = [f for f in files if f.split('/')[1].startswith(args.subdir_suffix)]
        print(f"Reading directory {args.base_dir}/**/{args.subdir_prefix}*/**, {len(files)} files found")
    else:
        files = glob.glob(f"{args.base_dir}/**/*_summary.csv", recursive=True)
        print(f"Reading directory {args.base_dir}, {len(files)} files found")
    
    assert len(files) > 0, "No files found, check the directory"
    summary_dfs = parse_summary_files(files, args.base_dir)

    current_date = datetime.date.today()

    if args.out_prefix:
        summary_dfs.to_csv(f'{args.base_dir}/{args.out_prefix}.tsv', sep='\t', index=False)
    else:
        summary_dfs.to_csv(f'{args.base_dir}/performance_all_{current_date}.tsv', sep='\t', index=False)
    print(f"Saved to {args.base_dir}/performance_all_{current_date}.tsv")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--base_dir', default=os.getcwd(), type=str, help='Base directory of the experiment')
    parser.add_argument('--subdir_prefix', default='', type=str, help='Find subdirs that startwith the suffix')
    parser.add_argument('--out_prefix', default='', type=str, help='Output file prefix')
    
    args = parser.parse_args()
    main(args)