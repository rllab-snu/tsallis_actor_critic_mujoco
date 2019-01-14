export CUDA_VISIBLE_DEVICES=0
python -m spinup.run sac --env MountainCarContinuous-v0 --exp_name mountaincar_sac --alpha 0.2 --seed 0 10 20 30 40 50 60 70 80 90 &

#export CUDA_VISIBLE_DEVICES=1
#python -m spinup.run td3 --env MountainCarContinuous-v0 --exp_name mountaincar_td3 --seed 0 10 20 30 40 50 60 70 80 90 &

#export CUDA_VISIBLE_DEVICES=2
#python -m spinup.run ppo --env MountainCarContinuous-v0 --exp_name mountaincar_ppo --seed 0 10 20 30 40 50 60 70 80 90 

#export CUDA_VISIBLE_DEVICES=3
#python -m spinup.run trpo --env MountainCarContinuous-v0 --exp_name mountaincar_trpo --seed 0 10 20 30 40 50 60 70 80 90 &

#export CUDA_VISIBLE_DEVICES=0
#python -m spinup.run ddpg --env MountainCarContinuous-v0 --exp_name mountaincar_ddpg --seed 0 10 20 30 40 50 60 70 80 90 &
