# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/record_base_u.csv --denoise_loss yes --orth_loss no --att no> logs/noisy/base_u.py 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/record_base_o_u.csv --denoise_loss yes --orth_loss yes --att no> logs/noisy/base_o_u.py 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/record_star_u1.csv --denoise_loss yes --orth_loss no --att no --backbone star> logs/noisy/star_u1.py 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/record_star_o_u2.csv --denoise_loss yes --orth_loss yes --att no --backbone star> logs/noisy/star_o_u2.py 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/record_star_o_u1.csv --denoise_loss yes --orth_loss yes --att no --backbone star> logs/noisy/star_o_u1.py 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/record_star_u_att1.csv --denoise_loss yes --orth_loss no --att yes --backbone star> logs/noisy/star_u_att1.py 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/record_star_o_u_att1.csv --denoise_loss yes --orth_loss yes --att yes --backbone star> logs/noisy/star_o_u_att1.py 2>&1 &

# google
CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/my_star_o_u.csv --denoise_loss yes --orth_loss yes --att no --backbone star --ptname my> logs/noisy/star_o_u.py 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/record_star_u2.csv --denoise_loss yes --orth_loss no --att no --backbone star> logs/noisy/star_u2.py 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/record_star_o2.csv --denoise_loss no --orth_loss yes --att no --backbone star> logs/noisy/star_o2.py 2>&1 &

# lege test
# lege test
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/lege_star_o_u1.csv --denoise_loss yes --orth_loss yes --att no --backbone star --dataset lege> logs/noisy/lege_star_o_u.py 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/lege_star_o_u2.csv --denoise_loss yes --orth_loss yes --att no --backbone star --dataset lege> logs/noisy_lege_star_o_u_2.py 2>&1 &
# 如果要测试 乐歌数据集 --dataset lege