export CUDA_VISIBLE_DEVICES=0
python -m spinup.run aeis --env Ant-v2 --exp_name ant_aeis --seed 0 10 20 30 40 50 60 70 80 90 &

export CUDA_VISIBLE_DEVICES=1
python -m spinup.run aeis --env Hopper-v2 --exp_name hopper_aeis --seed 0 10 20 30 40 50 60 70 80 90 

export CUDA_VISIBLE_DEVICES=0
python -m spinup.run aeis --env Swimmer-v2 --exp_name swim_aeis --seed 0 10 20 30 40 50 60 70 80 90 &

export CUDA_VISIBLE_DEVICES=1
python -m spinup.run aeis --env Pusher-v2 --exp_name push_aeis --seed 0 10 20 30 40 50 60 70 80 90 

export CUDA_VISIBLE_DEVICES=0
python -m spinup.run aeis --env Humanoid-v2 --exp_name human_aeis --seed 0 10 20 30 40 50 60 70 80 90 &
