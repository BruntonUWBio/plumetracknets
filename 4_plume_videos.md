## Some commands for generating animations
* Assumes that you have already completed [Agent evaluation](2_plume_eval.md) and [Figure/Report generation](1_plume_report.md) 


### Generate vids and plots (ppo/postEvalCli.py)
````bash
MODELDIRS=$(ls -d /home/satsingh/plume/plumezoo/latest/fly/memory/*VRNN*/)
echo $MODELDIRS
for DIR in $MODELDIRS; do
    LOGFILE=${DIR}/posteval.log
    python -u postEvalCli.py --model_dir $DIR \
      --viz_episodes 10 >> $LOGFILE 2>&1 &
done
tail -f /home/satsingh/plume/plumezoo/latest/fly/memory/*VRNN*/posteval.log


# Sparse
for DIR in $MODELDIRS; do
    LOGFILE=${DIR}/posteval.log
    python -u postEvalCli.py --model_dir $DIR \
      --viz_episodes 10 --birthxs 0.4 >> $LOGFILE 2>&1 &
done
tail -f /home/satsingh/plume/plumezoo/latest/fly/memory/*VRNN*/posteval.log


# Stitch videos side-by-side: see vid_stitch_cli for more options
for FNEURAL in $(find /home/satsingh/plume/plumezoo/latest/fly/memory/ -name "*pca3d_common_ep*.mp4"); do 
    FTRAJ=$(echo $FNEURAL | sed s/_pca3d_common//g) 
    # echo $FNEURAL $FTRAJ
    python -u ~/plume/plume2/vid_stitch_cli.py --fneural $FNEURAL --ftraj $FTRAJ
done
````
