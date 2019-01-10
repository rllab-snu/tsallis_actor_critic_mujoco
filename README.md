# Tsallis Actor Critic
## Run single experiment
```sh
cd tsallis_actor_critic_mujoco
python -m spinup.run tac --env HalfCheetah-v2 --exp_name half_tac_alpha_cst_q_1.5_cst_gaussian_q_log --epochs 200 --lr 1e-3 --q 1.5 --pdf_type gaussian --log_type q-log --alpha_schedule constant --q_schedule constant --seed 0 10 20 30 40 50 60 70 80 90
```
## Run multiple experiments
```sh
cd tsallis_actor_critic_mujoco
./tsallis_half_cheetah.sh
```
