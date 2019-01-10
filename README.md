# Tsallis Actor Critic
## Installaction
### Prerequisite
```sh
sudo apt-get update && sudo apt-get install libopenmpi-dev
```
### Install MuJoCo
pip install gym[mujoco,robotics]

### Install Tsallis Actor Critic
```sh
cd tsallis_actor_critic_mujoco
pip install -e .
```

## Run single experiment
```sh
cd tsallis_actor_critic_mujoco
python -m spinup.run tac --env HalfCheetah-v2 --exp_name half_tac_alpha_cst_q_1.5_cst_gaussian_q_log  --epochs 200 --lr 1e-3 --q 1.5 --pdf_type gaussian --log_type q-log --alpha_schedule constant --q_schedule constant --seed 0 10 20 30 40 50 60 70 80 90
```
Results will be saved in _data_ folder

### Experiment naming convention (Recommendation)
[env]\_[algorithm]\_alpha\_[alpha_schedule]\_q\_[entropic_index]\_[q_schedule]\_[distribution]\_[entropy_type]
- [env]: Environment name, ex) half
- [algorithm]: Algorithm name, ex) tac
- [alpha_schedule] indicates _alpha_schedule_. Use cst for constant and sch for scheduling
- [entropic_index] indicates _q_
- [q_schedule] is _q_schedule_. Use cst for constant and sch for scheduling
- [distribution] indicates _pdf\_type_ which has two options: Gaussian and $q$-Gaussian.
- [entropy_type] indicates _log\_type_ which has two options: $\log$ and $q-\log$

## Run multiple experiments
```sh
cd tsallis_actor_critic_mujoco
./tsallis_half_cheetah.sh
```
