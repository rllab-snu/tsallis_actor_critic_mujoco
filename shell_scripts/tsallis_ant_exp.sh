export CUDA_VISIBLE_DEVICES=0
python -m spinup.run tac --env Ant-v2 --exp_name ant_tac_alpha_cst_q_0.5_cst_gaussian_q_log --epochs 200 --lr 1e-3 --q 0.5 --pdf_type gaussian --log_type q-log --alpha_schedule constant --q_schedule constant --seed 0 10 20 30 40 50 60 70 80 90 &

export CUDA_VISIBLE_DEVICES=0
python -m spinup.run tac --env Ant-v2 --exp_name ant_tac_alpha_cst_q_0.7_cst_gaussian_q_log --epochs 200 --lr 1e-3 --q 0.7 --pdf_type gaussian --log_type q-log --alpha_schedule constant --q_schedule constant --seed 0 10 20 30 40 50 60 70 80 90 &
 
export CUDA_VISIBLE_DEVICES=0
python -m spinup.run tac --env Ant-v2 --exp_name ant_tac_alpha_cst_q_1.2_cst_gaussian_q_log --epochs 200 --lr 1e-3 --q 1.2 --pdf_type gaussian --log_type q-log --alpha_schedule constant --q_schedule constant --seed 0 10 20 30 40 50 60 70 80 90 &
 
export CUDA_VISIBLE_DEVICES=0
python -m spinup.run tac --env Ant-v2 --exp_name ant_tac_alpha_cst_q_1.5_cst_gaussian_q_log --epochs 200 --lr 1e-3 --q 1.5 --pdf_type gaussian --log_type q-log --alpha_schedule constant --q_schedule constant --seed 0 10 20 30 40 50 60 70 80 90 &
    
export CUDA_VISIBLE_DEVICES=1
python -m spinup.run tac --env Ant-v2 --exp_name ant_tac_alpha_cst_q_1.7_cst_gaussian_q_log --epochs 200 --lr 1e-3 --q 1.7 --pdf_type gaussian --log_type q-log --alpha_schedule constant --q_schedule constant --seed 0 10 20 30 40 50 60 70 80 90 &

export CUDA_VISIBLE_DEVICES=0
python -m spinup.run tac --env Ant-v2 --exp_name ant_tac_alpha_cst_q_2.0_cst_gaussian_q_log --epochs 200 --lr 1e-3 --q 2.0 --pdf_type gaussian --log_type q-log --alpha_schedule constant --q_schedule constant --seed 0 10 20 30 40 50 60 70 80 90 

export CUDA_VISIBLE_DEVICES=1
python -m spinup.run tac --env Ant-v2 --exp_name ant_tac_alpha_cst_q_3.0_cst_gaussian_q_log --epochs 200 --lr 1e-3 --q 3.0 --pdf_type gaussian --log_type q-log --alpha_schedule constant --q_schedule constant --seed 0 10 20 30 40 50 60 70 80 90 &

export CUDA_VISIBLE_DEVICES=0
python -m spinup.run tac --env Ant-v2 --exp_name ant_tac_alpha_cst_q_5.0_cst_gaussian_q_log --epochs 200 --lr 1e-3 --q 5.0 --pdf_type gaussian --log_type q-log --alpha_schedule constant --q_schedule constant --seed 0 10 20 30 40 50 60 70 80 90 
