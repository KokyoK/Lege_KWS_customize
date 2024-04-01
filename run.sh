# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py > logs/log_orth.py 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py > logs/log_noise_base2.py 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py > logs/log_noise_orth2.py 2>&1 &