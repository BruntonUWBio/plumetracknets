
# Instructions/Scripts to generate the images used in the manuscript from agent evaluation data

Please perform these steps in order
* Be sure to have run the [Prerequisites](0_plume_prereqs.md)
* Eval logs --> report dataframes (intermediate files)
* Generate reports i.e. plots used in figure
* Copy (per-agent) reports to central location

## Eval logs --> report dataframes (intermediate files)
````bash
cd ~/plume/plume2/ppo/
jupyter-nbconvert log2episodes.ipynb --to python
BASEDIR=~/plume/plumezoo/latest/fly/memory/

LOGFILES=$(find ${BASEDIR} -name "*.pkl")
MAXJOBS=24
for LOGFILE in $LOGFILES; do
  while (( $(jobs -p | wc -l) >= $MAXJOBS )); do sleep 1; done
  python -u log2episodes.py --logfile $LOGFILE >> /dev/null 2>&1 &
done
````

## Generate reports i.e. plots used in figure 
````bash
## Subspace
cd ~/plume/plume2/ppo/
jupyter-nbconvert report_common_subspace.ipynb --to python 
jupyter-nbconvert report_ttcs.ipynb --to python 

BASEDIR=~/plume/plumezoo/latest/fly/memory/
MODEL_FNAMES=$(find $BASEDIR -name "*.pt" | grep VRNN)
MAXJOBS=24
for FNAME in $MODEL_FNAMES; do
 while (( $(jobs -p | wc -l) >= $MAXJOBS )); do sleep 10; done
 MDIR=${FNAME%.pt}/
 python -u report_common_subspace.py --model_fname $FNAME > $MDIR/report_comsub.log 2>&1 &
 python -u report_ttcs.py --model_fname $FNAME > $MDIR/report_ttcs.log 2>&1 &
done
tail -f ${BASEDIR}/*VRNN*/report_comsub.log ${BASEDIR}/*VRNN*/report_ttcs.log


## Regime dists
cd ~/plume/plume2/ppo/
jupyter-nbconvert report_regime_dists.ipynb --to python 

BASEDIR=~/plume/plumezoo/latest/fly/memory/
MODEL_FNAMES=$(find $BASEDIR -name "*.pt" | grep VRNN)
MAXJOBS=24
for FNAME in $MODEL_FNAMES; do
 while (( $(jobs -p | wc -l) >= $MAXJOBS )); do sleep 10; done
 MDIR=${FNAME%.pt}/
 python -u report_regime_dists.py --model_fname $FNAME > $MDIR/report_regime.log 2>&1 &
done
tail -f ${BASEDIR}/*VRNN*/report_regime.log


## Regression R2s
BASEDIR=~/plume/plumezoo/latest/fly/memory/
cd ~/plume/plume2/ppo/
jupyter-nbconvert report_correlations.ipynb --to python

MODELDIRS=$(ls -d ${BASEDIR}/*VRNN*/)
MAXJOBS=24
for MDIR in $MODELDIRS; do
  while (( $(jobs -p | wc -l) >= $MAXJOBS )); do sleep 10; done
  python -u report_correlations.py --model_dir $MDIR > $MDIR/report_correlations.log 2>&1 &
done
tail -f ${BASEDIR}/*VRNN*/report_correlations.log
cat ${BASEDIR}/*VRNN*/report_correlations.log | grep "test data"
cat ${BASEDIR}/*VRNN*/report_arch/*.json 


# Arch
BASEDIR=~/plume/plumezoo/latest/fly/memory/
cd ~/plume/plume2/ppo/
jupyter-nbconvert report_arch.ipynb --to python 

BASEDIR=~/plume/plumezoo/latest/fly/memory/
MODELDIRS=$(ls -d ${BASEDIR}/*VRNN*/)
MAXJOBS=24
for MDIR in $MODELDIRS; do
  while (( $(jobs -p | wc -l) >= $MAXJOBS )); do sleep 10; done
  python -u report_arch.py --model_dir $MDIR > $MDIR/report_arch.log 2>&1 &
done
tail -f ${BASEDIR}/*VRNN*/report_arch.log
````

## Copy (per-agent) reports to central location
````bash
ZOODIR=~/walle/plume/plumezoo/latest/fly/memory/
REPORTDIR=~/PlumeTrackNets/

\cp ${ZOODIR}/plume_*VRNN*/report_arch/R2s_common_*.png ${REPORTDIR}/
\cp ${ZOODIR}/plume_*VRNN*/report_arch/repr*.png ${REPORTDIR}/
\cp ${ZOODIR}/plume_*VRNN*/report_arch/timescales_*.png ${REPORTDIR}/
\cp ${ZOODIR}/plume_*VRNN*/report_arch/eigenspectra_*.png ${REPORTDIR}/
\cp ${ZOODIR}/plume_*VRNN*/report_arch/*.png ${REPORTDIR}/

\cp ${ZOODIR}/plume_*VRNN*/report_common_subspace/comsub_regime*.png ${REPORTDIR}/
\cp ${ZOODIR}/plume_*VRNN*/report_common_subspace/comsub_odor_lastenc_*.png ${REPORTDIR}/
\cp ${ZOODIR}/plume_*VRNN*/report_common_subspace/comsub_odor_ma_*.png ${REPORTDIR}/
\cp ${ZOODIR}/plume_*VRNN*/report_common_subspace/comsub_odor_ewm_*.png ${REPORTDIR}/
\cp ${ZOODIR}/plume_*VRNN*/report_common_subspace/comsub_odor_enc_*.png ${REPORTDIR}/
\cp ${ZOODIR}/plume_*VRNN*/report_common_subspace/comsub_agent_angle_ground*.png ${REPORTDIR}/
\cp ${ZOODIR}/plume_*VRNN*/report_common_subspace/scree_*.png ${REPORTDIR}/

\cp ${ZOODIR}/plume_*VRNN*/report_common_subspace/ttcs_*.png ${REPORTDIR}/
\cp ${ZOODIR}/plume_*VRNN*/report_common_subspace/comsub_by_centroid*.png ${REPORTDIR}/

\cp ${ZOODIR}/report_diffs/home_by_arch_facet.png ${REPORTDIR}/ # Fig 4
\cp ${ZOODIR}/report_diffs/start_stop*.png ${REPORTDIR}/ # not used
\cp ${ZOODIR}/report_diffs/home_by_arch_*.png ${REPORTDIR}/ # not used
\cp ${ZOODIR}/report_diffs/home_by_dataset_*.png ${REPORTDIR}/ # not used

\cp ${ZOODIR}/plume_*VRNN*/report_regime_dists/regime_histos_*.png ${REPORTDIR}/
\cp ${ZOODIR}/plume_*VRNN*/report_regime_dists/regime_dists_*_TRACK_CD.png ${REPORTDIR}/
\cp ${ZOODIR}/plume_*VRNN*/report_common_subspace/limitcycle_*.png ${REPORTDIR}/

\cp ${ZOODIR}/report_arch/*.png ${REPORTDIR}/ # eigen, timescale, correlation R^2s

\cp ${ZOODIR}/report_common_subspace/regime_traj_*_OOB*.png ${REPORTDIR}/ 
\cp ${ZOODIR}/report_common_subspace/regime_traj_*_HOME*.png ${REPORTDIR}/ 
\cp ${ZOODIR}/report_common_subspace/regime_neural_*_OOB*.png ${REPORTDIR}/ 
\cp ${ZOODIR}/report_common_subspace/regime_neural_*_HOME*.png ${REPORTDIR}/ 
````

## View
* View outputs in ```REPORTDIR```
* Run ```ppo/report_plots.ipynb``` for some other figures
* Above workflow does all plotting on the command-line; you can also run the individual ipynbs if you prefer that.
