export CUDA_VISIBLE_DEVICES=0
python -m spinup.run tac --env Walker2d-v2 --exp_name walker_tac_q_1.0 --q 1.0 --seed 0 10 20 30 40 50 60 70 80 90 &

export CUDA_VISIBLE_DEVICES=1
python -m spinup.run tac --env Walker2d-v2 --exp_name walker_tac_q_1.5 --q 1.5 --seed 0 10 20 30 40 50 60 70 80 90 &

export CUDA_VISIBLE_DEVICES=2
python -m spinup.run tac --env Walker2d-v2 --exp_name walker_tac_q_2.0 --q 2.0 --seed 0 10 20 30 40 50 60 70 80 90 &

export CUDA_VISIBLE_DEVICES=3
python -m spinup.run tac --env Walker2d-v2 --exp_name walker_tac_q_2.5 --q 2.5 --seed 0 10 20 30 40 50 60 70 80 90 

export CUDA_VISIBLE_DEVICES=0
python -m spinup.run tac --env Walker2d-v2 --exp_name walker_tac_q_3.0 --q 3.0 --seed 0 10 20 30 40 50 60 70 80 90 &

export CUDA_VISIBLE_DEVICES=1
python -m spinup.run tac --env Walker2d-v2 --exp_name walker_tac_q_5.0 --q 5.0 --seed 0 10 20 30 40 50 60 70 80 90 &

export CUDA_VISIBLE_DEVICES=2
python -m spinup.run tac --env Walker2d-v2 --exp_name walker_tac_q_10.0 --q 10.0 --seed 0 10 20 30 40 50 60 70 80 90 &
