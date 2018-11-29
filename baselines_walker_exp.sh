export CUDA_VISIBLE_DEVICES=0
python -m spinup.run sac --env Walker2d-v2 --exp_name walker_sac --alpha 0.2 --seed 0 10 20 30 40 50 60 70 80 90 &

export CUDA_VISIBLE_DEVICES=1
python -m spinup.run td3 --env Walker2d-v2 --exp_name walker_td3 --seed 0 10 20 30 40 50 60 70 80 90 &

export CUDA_VISIBLE_DEVICES=2
python -m spinup.run ppo --env Walker2d-v2 --exp_name walker_ppo --seed 0 10 20 30 40 50 60 70 80 90 

export CUDA_VISIBLE_DEVICES=3
python -m spinup.run trpo --env Walker2d-v2 --exp_name walker_trpo --seed 0 10 20 30 40 50 60 70 80 90 &

export CUDA_VISIBLE_DEVICES=0
python -m spinup.run ddpg --env Walker2d-v2 --exp_name walker_ddpg --seed 0 10 20 30 40 50 60 70 80 90 &