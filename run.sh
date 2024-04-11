# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --k 2 --s 1 --log logs/base_k2.csv > logs/base_log.py 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --k 3 --s 1 --log logs/base_k3.csv > logs/base_log2.py 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --k 2 --s 1 --log logs/k2.csv > logs/log.py 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --k 3 --s 1 --log logs/k3.csv > logs/log2.py 2>&1 &