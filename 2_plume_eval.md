# Instructions/Scripts to (re)generate agent evaluation data i.e. the "behavioral assays". Only required if not using the downloaded data

Please perform these steps in order
* Be sure to have run the [Prerequisites](0_plume_prereqs.md)
* Run evaluation on all models in PLUMEZOO i.e. 14 RNNs and 14 MLPs x (2, 4, ... 12) timesteps of memory
* Select top-5 of each using tabulate.ipynb and either delete the rest or move these to a separate folder


## Run evaluation ("behavior assay" in paper)
````bash
ZOODIR=~/plume/plumezoo/latest/fly/memory/
FNAMES=$(find . -name "*.pt")
echo $FNAMES

MAXJOBS=24
DATASETS="constantx5b5 switch45x5b5 noisy3x5b5"
for DATASET in $DATASETS; do
for FNAME in $FNAMES; do
    while (( $(jobs -p | wc -l) >= MAXJOBS )); do echo "Sleeping..."; sleep 10; done 

    LOGFILE=$(echo $FNAME | sed s/.pt/_${DATASET}.evallog/g)
    SPARSE_MODIFIER=""
    if [[ $DATASET == "constantx5b5" ]]; then 
    	SPARSE_MODIFIER="--test_sparsity"
    fi
    nice python -u ~/plume/plume2/ppo/evalCli.py \
        --dataset $DATASET \
        --fixed_eval $SPARSE_MODIFIER \
        --viz_episodes 20 \
        --model_fname $FNAME >> $LOGFILE 2>&1 &
 done
done

tail -f *.evallog
````

## Command line summary of eval logs

````bash
cd $PLUMEZOO
for DIR in $(ls -d plume*/); do 
  C2=$(grep HOME $DIR/constantx5b5_summary.csv | wc -l)
  C3=$(grep HOME $DIR/switch45x5b5_summary.csv | wc -l)
  C4=$(grep HOME $DIR/noisy3x5b5_summary.csv | wc -l)
  C1=$(($C2 + $C3 + C4))  
  echo $C1 $C2 $C3 $C4 $DIR
done | sort -n
````

## Select top-5 of each arch
* Use ```code/tabulate.ipynb``` and either delete the rest or move these to a separate folder
* Once you've run the agent evaluations, you can now proceed to [Figure/Report generation](1_plume_report.md) 