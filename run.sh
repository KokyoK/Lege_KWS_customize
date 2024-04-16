# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/record_base_u.csv --denoise_loss yes --orth_loss no --att no> logs/noisy/base_u.py 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/record_base_o_u.csv --denoise_loss yes --orth_loss yes --att no> logs/noisy/base_o_u.py 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/record_star_u.csv --denoise_loss yes --orth_loss no --att no --backbone star> logs/noisy/star_u.py 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/record_star_o_u.csv --denoise_loss yes --orth_loss yes --att no --backbone star> logs/noisy/star_o_u.py 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/record_star_u_att.csv --denoise_loss yes --orth_loss no --att yes --backbone star> logs/noisy/star_u.py 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/record_star_o_u_att.csv --denoise_loss yes --orth_loss yes --att yes --backbone star> logs/noisy/star_o_u_att.py 2>&1 &