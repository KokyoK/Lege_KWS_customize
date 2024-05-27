# runme
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/myspec.csv --feat spec --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net specu --ptname my_spec> logs/noisy/myspec.py 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/myspec2.csv --feat spec --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net specu --ptname myspec2> logs/noisy/myspec.py 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/myspec2.csv --feat spec --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net specu --ptname myspec2> logs/noisy/myspec.py 2>&1 &


# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/noix.csv --feat spec --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net specu --ptname noix> logs/noisy/noix.py 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/noix2.csv --feat spec --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net specu --ptname noix2> logs/noisy/noix2.py 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/noix3.csv --feat spec --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net specu --ptname noix3> logs/noisy/noix3.py 2>&1 &



CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/oht.csv --feat spec --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net specu --ptname oht> logs/noisy/oht.py 2>&1 & # no cross att
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/oht2.csv --feat spec --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net specu --ptname oht2> logs/noisy/oht2.py 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/oh2.csv --feat spec --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net specu --ptname oh2> logs/noisy/oh2.py 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/oh3.csv --feat spec --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net specu --ptname oh3> logs/noisy/oh3.py 2>&1 &    # concat channel
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/oh4.csv --feat spec --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net specu --ptname oh4> logs/noisy/oh4.py 2>&1 &    # concat
# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/oh5.csv --feat spec --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net specu --ptname oh5> logs/noisy/oh5.py 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/oh5t.csv --feat spec --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net specu --ptname oh5t> logs/noisy/oh5t.py 2>&1 &