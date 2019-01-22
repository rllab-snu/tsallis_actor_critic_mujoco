# Tsallis Actor Critic
This repository provides the implementation of Tsallis actor critic (TAC) method based on Spinningup packages which is educational resource produced by OpenAI. We implemented TAC based on Spinningup packages https://github.com/openai/spinningup. TAC generalizes the standard Shannon-Gibbs entropy maximization in RL to the Tsallis entropy (https://en.wikipedia.org/wiki/Tsallis_entropy).
## Installaction
### Prerequisite
```sh
sudo apt-get update && sudo apt-get install libopenmpi-dev
```
### Install MuJoCo (Recommend)
```sh
pip install gym[mujoco,robotics]
```
### Install Spinningup with Tsallis Actor Critic
```sh
cd tsallis_actor_critic_mujoco
pip install -e .
```
### Install Custom Gym
```sh
cd tsallis_actor_critic_mujodo/custom_gym/
pip install -e .
```
If you want to add a customized environment, see https://github.com/openai/gym/tree/master/gym/envs#how-to-create-new-environments-for-gym

## Jupyter Notebook Examples for Tsallis Entropy and Dynamic Programming
```sh
cd tsallis_actor_critic_mujoco
cd spinup/algos/tac
ls
```
The following files will be shown
```
tac
├── core.py
├── tac.py
├── tf_tsallis_statistics.py
├── Example_Tsallis_MDPs.ipynb 
└── Example_Tsallis_statistics.ipynb
```
- Example_Tsallis_MDPs.ipynb shows the figure of performance error bound.
- Example_Tsallis_statistics.ipynb shows the multi armed bandit with maximum Tsallis entropy examples.

## Reproducing experiments
### Run test
```sh
cd tsallis_actor_critic_mujoco
python -m spinup.run tac --env HalfCheetah-v2
```

### Run single experiment
```sh
cd tsallis_actor_critic_mujoco
python -m spinup.run tac --env HalfCheetah-v2 --exp_name half_tac_alpha_cst_q_1.5_cst_gaussian_q_log  --epochs 200 --lr 1e-3 --q 1.5 --pdf_type gaussian --log_type q-log --alpha_schedule constant --q_schedule constant --seed 0 10 20 30 40 50 60 70 80 90
```
Results will be saved in _data_ folder

### Experiment naming convention (Recommend)
[env]\_[algorithm]\_alpha\_[alpha_schedule]\_q\_[entropic_index]\_[q_schedule]\_[distribution]\_[entropy_type]
- [env]: Environment name, ex) half
- [algorithm]: Algorithm name, ex) tac
- [alpha_schedule] indicates _alpha_schedule_. Use cst for constant and sch for scheduling
- [entropic_index] indicates _q_
- [q_schedule] is _q_schedule_. Use cst for constant and sch for scheduling
- [distribution] indicates _pdf\_type_ which has two options: _gaussian_ and _q-gaussian_
- [entropy_type] indicates _log\_type_ which has two options: _log_ and _q-log_

This convention will help you not forget a parameter setting.

### Run multiple experiments
```sh
cd tsallis_actor_critic_mujoco
./shell_scripts/tsallis_half_cheetah.sh
```
To run mulitple experiments at once, we employ a simple and easy way as follows:
```
run program_1 & program_2 & ... & program_n
```
