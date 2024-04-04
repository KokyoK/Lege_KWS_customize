# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py > logs/log_orth.py 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py > logs/log_no_denoise_loss.py 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py > logs/log_denoise_orth.py 2>&1 &