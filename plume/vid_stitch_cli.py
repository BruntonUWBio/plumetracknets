"""
# EvalCli
VIDS=$(find . -name "*pca*.mp4")

VIDS=$(find /home/satsingh/plume/plumezoo/latest/fly/memory/*VRNN*/ -name "*pca3d_ep*.mp4")
echo $VIDS


VIDS=$(find /home/satsingh/plume/plumezoo/latest/fly/memory/*VRNN*/ -name "*pca3d_ep*.mp4")
for VID in $VIDS; do 
    #python -u ~/plume/plume2/vid_stitch_cli.py --fneural $VID
    VID_TRAJ=$(echo $VID | sed s/_pca3d//g) 
    echo $VID $VID_TRAJ
    python -u ~/plume/plume2/vid_stitch_cli.py --fneural $VID --ftraj $VID_TRAJ
done

# Test 2/3 panel
# PostEvalCli -- See Postevalcli for generation
BASEDIR=~/plume/plumezoo/latest/fly/memory/plume_20210601_VRNN_constantx5b5noisy3x5b5_stepoob_bx0.30.8_t10000004000000_q2.00.5_dmx0.80.8_dmn0.70.4_h64_wd0.0001_n4_codeVRNN_seed3307e9/
VID_ACT=${BASEDIR}/constantx5b5_pca3d_common_ep224.mp4
VID_TRAJ=${BASEDIR}/constantx5b5_ep224.mp4
VID_EIG=${BASEDIR}/constantx5b5_ep224_eig.mp4
ls $VID_TRAJ $VID_ACT $VID_EIG
python -u ~/plume/plume2/vid_stitch_cli.py --fneural $VID_ACT --ftraj $VID_TRAJ
python -u ~/plume/plume2/vid_stitch_cli.py --fneural $VID_ACT --ftraj $VID_TRAJ --feig $VID_EIG


# 2 panel
VIDS=$(find . -name "*_0.4_*pca3d_common_ep*.mp4") # if in the VRNN dir
VIDS=$(find . -name "*pca3d_common_ep*.mp4") # if in the VRNN dir
cd /home/satsingh/plume/plumezoo/latest/fly/memory/
VIDS=$(find *VRNN*/ -name "*pca3d_common_ep*.mp4")

MAXJOBS=24
for VID_PCA in $VIDS; do 
    while (( $(jobs -p | wc -l) >= MAXJOBS )); do echo "Sleeping..."; sleep 10; done 
    VID_TRAJ=$(echo $VID_PCA | sed s/_pca3d_common//g) 
    echo 
    echo $(ls $VID_PCA $VID_TRAJ $VID_EIG | wc -l) $VID_PCA $VID_TRAJ $VID_EIG 
    python -u ~/plume/plume2/vid_stitch_cli.py --fneural $VID_PCA --ftraj $VID_TRAJ &
done


# 3 panel
VIDS=$(find . -name "*pca3d_common_ep*.mp4") # if in the VRNN dir
cd /home/satsingh/plume/plumezoo/latest/fly/memory/
VIDS=$(find *VRNN*/ -name "*pca3d_common_ep*.mp4")

MAXJOBS=24
for VID_PCA in $VIDS; do 
    while (( $(jobs -p | wc -l) >= MAXJOBS )); do echo "Sleeping..."; sleep 10; done 
    VID_TRAJ=$(echo $VID_PCA | sed s/_pca3d_common//g) 
    VID_EIG=${VID_TRAJ%.mp4}_eig.mp4
    echo 
    echo $(ls $VID_PCA $VID_TRAJ $VID_EIG | wc -l) $VID_PCA $VID_TRAJ $VID_EIG 
    python -u ~/plume/plume2/vid_stitch_cli.py --fneural $VID_PCA --ftraj $VID_TRAJ --feig $VID_EIG &
done


"""
import os
import sys
import argparse
from moviepy.editor import *

def merge_trajectory_neural_clips(ftraj, fneural, feig, fsuffix):
    clip2 = VideoFileClip(fneural)
    # clip1 = VideoFileClip(ftraj).resize(height=clip2.size[1])
    clip1 = VideoFileClip(ftraj)
    clips = [[clip1, clip2]]
    if feig is not None:
        clip3 = VideoFileClip(feig).resize(height=clip2.size[1])
        clips = [[clip1, clip2, clip3]]
        fsuffix += "3p"        
    final = clips_array(clips, bg_color=(255,255,255))
        
    fname = fneural.replace('pca3d', fsuffix)
    print(fsuffix, fname)
    final.write_videofile(fname, fps=15, verbose=False, logger=None)
    # final.write_videofile(fname, fps=15)
    print(f'Saved: {fname}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('--fneural', default=None)
    parser.add_argument('--ftraj', default=None)
    parser.add_argument('--feig', default=None)
    parser.add_argument('--fsuffix', default='merged')
    args = parser.parse_args()
    print(args)

    if (args.fneural is None) or (not os.path.exists(args.fneural)):
        print("Missing:", args.fneural)
        sys.exit(-1)

    if (args.ftraj is None) or (not os.path.exists(args.ftraj)):
        print("Missing:", args.ftraj)
        sys.exit(-1)

    merge_trajectory_neural_clips(args.ftraj, args.fneural, args.feig, args.fsuffix)    
