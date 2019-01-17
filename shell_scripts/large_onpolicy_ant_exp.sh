export CUDA_VISIBLE_DEVICES=2
python -m spinup.run ppo --env Ant-v2 --exp_name ant_large_ppo --hid [300,300] --seed 0 10 20 30 40 50 60 70 80 90 &

export CUDA_VISIBLE_DEVICES=3
python -m spinup.run trpo --env Ant-v2 --exp_name ant_large_trpo --hid [300,300] --seed 0 10 20 30 40 50 60 70 80 90
