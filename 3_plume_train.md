# Instructions/Scripts to train agents from scratch
* Be sure to have run the [Prerequisites](0_plume_prereqs.md)
* Train 14 RNNs and 14 MLPs x (2, 4, ... 12) timesteps of memory to create PLUMEZOO (shown below)
* Proceed to [Agent evaluation](2_plume_eval.md)
* Once you've run the agent evaluations, you can now proceed to [Figure/Report generation](1_plume_report.md) 

## Important Notes
* All development was done on an Ubuntu Linux v20.04 workstation with Intel Core i9-9940X CPU and a TITAN RTX GPU.
* Each seed takes about 16 hours to train and evaluate, with MLP and RNN models using 1 and 4 cores in parallel respectively.
* Deep RL experiments, in general, are tough to reproduce. See:
  * [SO post on GPU nondeterminism](https://stackoverflow.com/questions/50744565/how-to-handle-non-determinism-when-training-on-a-gpu)
  * [Nagarajan et al, The Impact of Nondeterminism on Reproducibility in Deep Reinforcement Learning](https://openreview.net/pdf?id=S1e-OsZ4e7)
  * [Henderson et al, Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560)
* I have done my best to control for randomness, but there are many sources of stochasticity to control for -- simulation seeds, training initializations, agent network initialization, compounding behavioral differences between agents, etc. 
* However, I have run this code multiple times, and the results are qualitatively the same; quantitatively, there is large variability between agents, but selecting top-5 out of 14 agents for each architecture reduces that variability and gives us a good representative sample of each class. (Experiment with increasing the number of agents trained from 14 to higher for potentially more control on the variability.)



## RNN
````bash
NUMPROC=4 # Walle/Weekend
SHAPE="step oob"

BIRTHX="0.3 0.8" #  Original Config
DATASET="constantx5b5 noisy3x5b5"
STEPS="1000000 4000000"
VARX="2.0 0.5"
DMAX="0.8 0.8"
DMIN="0.7 0.4"

ALGO=ppo
HIDDEN=64
DECAY=0.0001

EXPT=ExptMemory$(date '+%Y%m%d')
SAVEDIR=./trained_models/${EXPT}/
mkdir -p $SAVEDIR

MAXJOBS=7
for SEED in $(seq 14); do  
  for RNNTYPE in VRNN; do
    while (( $(jobs -p | wc -l) >= MAXJOBS )); do sleep 10; done 
    SEED=$RANDOM

    DATASTRING=$(echo -e $DATASET | tr -d ' ')
    SHAPESTRING=$(echo -e $SHAPE | tr -d ' ')
    BXSTRING=$(echo -e $BIRTHX | tr -d ' ')
    TSTRING=$(echo -e $STEPS | tr -d ' ')
    QVARSTR=$(echo -e $VARX | tr -d ' ')
    DMAXSTR=$(echo -e $DMAX | tr -d ' ')
    DMINSTR=$(echo -e $DMIN | tr -d ' ')

    OUTSUFFIX=$(date '+%Y%m%d')_${RNNTYPE}_${DATASTRING}_${SHAPESTRING}_bx${BXSTRING}_t${TSTRING}_q${QVARSTR}_dmx${DMAXSTR}_dmn${DMINSTR}_h${HIDDEN}_wd${DECAY}_n${NUMPROC}_code${RNNTYPE}_seed${SEED}$(openssl rand -hex 1)
    echo $OUTSUFFIX

    nice python -u main.py --env-name plume \
      --recurrent-policy \
      --dataset $DATASET \
      --num-env-steps ${STEPS} \
      --birthx $BIRTHX  \
      --flipping True \
      --qvar $VARX \
      --save-dir $SAVEDIR \
      --log-interval 1 \
      --r_shaping $(echo -e $SHAPE) \
      --algo $ALGO \
      --seed ${SEED} \
      --squash_action True \
      --diff_max $DMAX \
      --diff_min $DMIN \
      --num-processes $NUMPROC \
      --num-mini-batch $NUMPROC \
      --odor_scaling True \
      --rnn_type ${RNNTYPE} \
      --hidden_size $HIDDEN \
      --weight_decay ${DECAY} \
      --use-gae --num-steps 2048 --lr 3e-4 --entropy-coef 0.005 --value-loss-coef 0.5 --ppo-epoch 10 --gamma 0.99 --gae-lambda 0.95 --use-linear-lr-decay \
      --outsuffix ${OUTSUFFIX} > ${SAVEDIR}/${OUTSUFFIX}.log 2>&1 &

      echo "Sleeping.."
      sleep 0.5

   done
  done  
 done
done

tail -f ${SAVEDIR}/*.log

````

### MLPs Serial execution: -- Memory
````bash
NUMPROC=1 # Mycroft Weeknight
SHAPE="step oob"
BIRTHX="0.3 1.0"
DATASET="constantx5b5 noisy3x5b5"
VARX="1.0 0.5"
STEPS="100000 2000000"
DMAX="0.8 0.8"
DMIN="0.8 0.4"

SHAPE="step oob"
BIRTHX="1.0"
DATASET="noisy3x5b5"
VARX="0.5"
STEPS="2000000"
DMAX="0.8"
DMIN="0.3"

ALGO=ppo
HIDDEN=64
DECAY=0.0001

EXPT=ExptMemory$(date '+%Y%m%d')
SAVEDIR=./trained_models/${EXPT}/
mkdir -p $SAVEDIR

MAXJOBS=28
for SEED in $(seq 14); do
  for STACK in 02 04 06 08 10 12; do
    while (( $(jobs -p | wc -l) >= MAXJOBS )); do sleep 10; done 
    SEED=$RANDOM
    SHAPESTRING=$(echo -e $SHAPE | tr -d ' ')
    DATASTRING=$(echo -e $DATASET | tr -d ' ')
    BXSTRING=$(echo -e $BIRTHX | tr -d ' ')
    TSTRING=$(echo -e $STEPS | tr -d ' ')
    QVARSTR=$(echo -e $VARX | tr -d ' ')
    DMAXSTR=$(echo -e $DMAX | tr -d ' ')
    DMINSTR=$(echo -e $DMIN | tr -d ' ')

    OUTSUFFIX=$(date '+%Y%m%d')_MLP_s${STACK}_${DATASTRING}_${SHAPESTRING}_bx${BXSTRING}_t${TSTRING}_q${QVARSTR}_dmx${DMAXSTR}_dmn${DMINSTR}_wd${DECAY}_n${NUMPROC}_codeMLPs${STACK}_seed${SEED}$(openssl rand -hex 1)

    echo $OUTSUFFIX

    nice python -u main.py --env-name plume \
      --stacking ${STACK} \
      --dataset $DATASET \
      --num-env-steps ${STEPS} \
      --birthx $BIRTHX  \
      --flipping True \
      --qvar $VARX \
      --save-dir $SAVEDIR \
      --log-interval 1 \
      --r_shaping $(echo -e $SHAPE) \
      --algo $ALGO \
      --seed ${SEED} \
      --squash_action True \
      --diff_max $DMAX \
      --diff_min $DMIN \
      --num-processes $NUMPROC \
      --num-mini-batch $NUMPROC \
      --odor_scaling True \
      --hidden_size $HIDDEN \
      --weight_decay ${DECAY} \
      --use-gae --num-steps 2048 --lr 3e-4 --entropy-coef 0.005 --value-loss-coef 0.5 --ppo-epoch 10 --gamma 0.99 --gae-lambda 0.95 --use-linear-lr-decay \
      --outsuffix ${OUTSUFFIX} > ${SAVEDIR}/${OUTSUFFIX}.log 2>&1 &
  done
 done
done

tail -f ${SAVEDIR}/*.log
````

