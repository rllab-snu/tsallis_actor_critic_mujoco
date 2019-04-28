export CUDA_VISIBLE_DEVICES=0
python -m spinup.run tac --env Pusher-v2 --exp_name push_tac_alpha_0.2_cst_q_2.0_lin_gaussian_q_log --epochs 200 --lr 1e-3 --alpha 0.2 --q 2.0 --pdf_type gaussian --log_type q-log --alpha_schedule constant --q_schedule linear --seed 0 10 20 30 40 50 60 70 80 90 &

export CUDA_VISIBLE_DEVICES=1
python -m spinup.run tac --env Pusher-v2 --exp_name push_tac_alpha_0.02_cst_q_2.0_lin_gaussian_q_log --epochs 200 --lr 1e-3 --alpha 0.02 --q 2.0 --pdf_type gaussian --log_type q-log --alpha_schedule constant --q_schedule linear --seed 0 10 20 30 40 50 60 70 80 90 

export CUDA_VISIBLE_DEVICES=0
python -m spinup.run tac --env Pusher-v2 --exp_name push_tac_alpha_0.002_cst_q_2.0_lin_gaussian_q_log --epochs 200 --lr 1e-3 --alpha 0.002 --q 2.0 --pdf_type gaussian --log_type q-log --alpha_schedule constant --q_schedule linear --seed 0 10 20 30 40 50 60 70 80 90 &

export CUDA_VISIBLE_DEVICES=1
python -m spinup.run tac --env Pusher-v2 --exp_name push_tac_alpha_0.5_cst_q_2.0_lin_gaussian_q_log --epochs 200 --lr 1e-3 --alpha 0.5 --q 2.0 --pdf_type gaussian --log_type q-log --alpha_schedule constant --q_schedule linear --seed 0 10 20 30 40 50 60 70 80 90 


