## Prerequisites: Packages needed, data organization, simulation data generation and configuration

### Code set up
* Instructions tested on an [Ubuntu 18.04](https://releases.ubuntu.com/18.04.4/) machine with [Anaconda Python 3](https://www.anaconda.com/products/individual):

* Git clone this repository

* (**See conda py39 note below**) If you'd like, set up and activate a [virtual environment](https://docs.python.org/3/tutorial/venv.html). For example:
```bash
python -m venv /path/to/myvenv
source /path/to/myvenv/bin/activate
``` 

* Open notebook files using [Jupyter Notebook](https://jupyter-notebook.readthedocs.io/en/stable/) (this comes with your Anaconda installation)


### Packages/environment

* Our work builds upon two well-documented open-source RL repos
  * [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
  * [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail)
To begin, please install these and all their required packages


**Dec 2023 Addendum:** Lots of dependency changes since the time of release of our code necessitates that you use a conda Python 3.9 environment like below
```bash
# https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-python.html
conda create -n py39 python=3.9
conda activate py39

conda install ffmpeg

# https://pypi.org/project/mpi4py/
conda install -c conda-forge mpi4py openmpi 

# Install gym==0.21.0, manually if needed
pip install gym==0.21.0 # likely to fail, if so, use below
# git clone https://github.com/openai/gym.git
# cd gym
# git checkout v0.21.0
# https://github.com/openai/gym/issues/3202
pip install git+https://github.com/openai/gym.git@9180d12e1b66e7e2a1a622614f787a6ec147ac40 


pip install -r requirements.txt

git clone https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail.git
python pytorch-a2c-ppo-acktr-gail/main.py --algo ppo --env-name "Pendulum-v1"
```

### Data

* Data (agent model/network files, model evaluation data) can be downloaded from Figshare: https://doi.org/10.6084/m9.figshare.16879539.v1 (approx. 9GB)

* ```PlumeTrackNets_20211026.zip.split.xx``` files on Figshare were created using the linux command ```split -n 9 PlumeTrackNets_20211026.zip``` 

* [Un-split the zip file](https://ostechnix.com/split-combine-files-command-line-linux/) and Unzip in a folder e.g. ```~/plume/plumezoo/```


### Configure paths

* Edit ```code/config.py``` to set paths, seeds, color-schemes, etc. 

* Some key paths that have been used in the rest of this documentation include: 
````bash
datadir = '/home/satsingh/plume/plumedata/' # where simulation outputs go
plumezoo = '~/plume/plumezoo/latest/fly/memory/' # where the agent models and eval. data go; downloaded zip file gets unzipped here
basedir = '/home/satsingh/plume/plume2/' # code base-folder; i.e. GIT_REPO/code/
````


### Simulation data pre-generation

* Using ```code/sim_cli.py```, run this to pre-generate plume simulations needed for manuscript. See code file for more examples:

````bash
for DATASET in noisy3 constant switch45; do
  echo $DATASET
  python -u sim_cli.py \
    --duration 120 \
    --dataset_name $DATASET \
    --wind_magnitude 0.5 \
    --birth_rate 1.0 \
    --fname_suffix x5b5 > ${DATASET}x5b5.log 2>&1 &
 done
done
tail -f *.log
````



