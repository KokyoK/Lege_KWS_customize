# runme
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/myspec.csv --feat spec --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net specu --ptname my_spec> logs/noisy/myspec.py 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/myspec2.csv --feat spec --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net specu --ptname myspec2> logs/noisy/myspec.py 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/myspec2.csv --feat spec --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net specu --ptname myspec2> logs/noisy/myspec.py 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/oh.csv --feat spec --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net specu --ptname oh> logs/noisy/oh.py 2>&1 &