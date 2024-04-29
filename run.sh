# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/record_base_u.csv --denoise_loss yes --orth_loss no --att no> logs/noisy/base_u.py 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/record_base_o_u.csv --denoise_loss yes --orth_loss yes --att no> logs/noisy/base_o_u.py 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/record_star_u.csv --denoise_loss yes --orth_loss no --att no --backbone star> logs/noisy/star_u.py 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/record_star_o_u.csv --denoise_loss yes --orth_loss yes --att no --backbone star> logs/noisy/star_o_u.py 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/record_star_u_att.csv --denoise_loss yes --orth_loss no --att yes --backbone star> logs/noisy/star_u.py 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/record_star_o_u_att.csv --denoise_loss yes --orth_loss yes --att yes --backbone star> logs/noisy/star_o_u_att.py 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/record_star_u_att.csv --denoise_loss yes --orth_loss no --att yes --backbone star> logs/noisy/star_u.py 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/record_star_o_u_sub.csv --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net sub> logs/noisy/star_o_u_att.py 2>&1 &



# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/lege_base.csv --denoise_loss no --orth_loss no --att no --backbone res --denoise_net sub --dataset lege > logs/noisy_lege/base.py 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/lege_star_o_u.csv --denoise_loss no --orth_loss yes --att no --backbone star --denoise_net unet --dataset lege> logs/noisy_lege/star_o_u.py 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/my_align_2.csv --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net unet --ptname my> logs/noisy/align.py 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/my_align_orth.csv --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net unet --p



tname my> logs/noisy/align.py 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/my_align_cov.csv --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net unet --ptname my> logs/noisy/aligncov.py 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/my_align_no_orth.csv --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net unet --ptname no_orth> logs/noisy/aligncov.py 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/my_star_plain.csv --denoise_loss no --orth_loss no --att no --backbone star --denoise_net unet --ptname star_plain> logs/noisy/align_nod.py 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/tcresnet8.csv --denoise_loss no --orth_loss no --att no --backbone res --denoise_net unet --ptname res> logs/noisy/res_nod.py 2>&1 &

##### BC resnet #####
# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/bc.csv --denoise_loss no --orth_loss no --att no --backbone bc --denoise_net unet --ptname bc_base> logs/noisy/aligncov.py 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/bco.csv --denoise_loss no --orth_loss yes --att no --backbone bc --denoise_net unet --ptname bco> logs/noisy/aligncov.py 2>&1 &