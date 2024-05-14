

#### my ###
# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/my.csv --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net unet --ptname my> logs/noisy/my.py 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/tune2.csv --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net unet --ptname my> logs/noisy/tune2.py 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/my2.csv --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net unet --ptname my> logs/noisy/aligncov.py 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/my_align_no_orth.csv --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net unet --ptname no_orth> logs/noisy/aligncov.py 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/my_tune.csv --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net specu --ptname tune> logs/noisy/align_nod.py 2>&1 &

# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/tcresnet8.csv --denoise_loss no --orth_loss no --att no --backbone res --denoise_net unet --ptname res> logs/noisy/res_nod.py 2>&1 &

##### TC 8 #####
CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/tcx.csv --denoise_loss no --orth_loss no --att no --feat mfcc  --backbone res --denoise_net unet --ptname tcx> logs/noisy/tc.py 2>&1 &
##### TC 14 ####
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/tc14.csv --denoise_loss no --orth_loss no --att no --backbone tc14 --denoise_net unet --ptname tc14> logs/noisy/tc14.py 2>&1 &

##### BC resnet #####
CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/bcx.csv --denoise_loss no --orth_loss no --att no --backbone bc --denoise_net unet --ptname bcx> logs/noisy/bc.py 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/bco.csv --denoise_loss no --orth_loss yes --att no --backbone bc --denoise_net unet --ptname bco> logs/noisy/aligncov.py 2>&1 &

##### Decouple Net #####
CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/decouplex.csv --denoise_loss no --orth_loss no --feat mfcc --att no --backbone decouple --denoise_net unet --ptname dmfccx> logs/noisy/decouple_mfcc.py 2>&1 &

# camm
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/specu.csv --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net specu --ptname cammd> logs/noisy/spec.py 2>&1 &

# single
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/onlykw.csv --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net specu --ptname cammd> logs/noisy/spec.py 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/onlysv.csv --denoise_loss yes --orth_loss yes --att no --backbone star --denoise_net specu --ptname cammd> logs/noisy/spec.py 2>&1 &

# LSN
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/specu.csv --denoise_loss no --orth_loss no --att no --backbone star --denoise_net specu --ptname star> logs/noisy/star.py 2>&1 &

# LSN + MM
# CUDA_VISIBLE_DEVICES=0 nohup python nn_main.py --log logs/LSNMM.csv --denoise_loss no --orth_loss yes --att no --backbone star --denoise_net specu --ptname star> logs/noisy/star.py 2>&1 &

# LSN + SUB
# CUDA_VISIBLE_DEVICES=1 nohup python nn_main.py --log logs/LSNSUB.csv --denoise_loss yes --orth_loss no --att no --backbone star --denoise_net specu --ptname star> logs/noisy/star.py 2>&1 &