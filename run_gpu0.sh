dataset='PACS' # office-home of PACS
algorithm=('MLDG' 'ERM' 'DANN' 'RSC' 'Mixup' 'MMD' 'CORAL' 'VREx') 
test_envs=2
gpu_ids=0
# data_dir='/home/data/PACS/'
data_dir='/home/data/OOD_dataset/PACS/'
max_epoch=240
net='resnet18'
task='img_dg'
output='./out'
erm_noise_path='/home/jcjiang/MM2024/model_dir/random/noise/ERM/2024-03-09_16-25-17/model.pkl'
erm_path='/home/jcjiang/MM2024/model_dir/random/ERM/2024-03-09_16-24-56/model.pkl'
andmask_noise_path='/home/jcjiang/MM2024/model_dir/random/noise/ANDMask/2024-03-09_16-25-17/model.pkl'
andmask_path='/home/jcjiang/MM2024/model_dir/random/ANDMask/2024-03-09_16-24-56/model.pkl'
mixup_noise_path='/home/jcjiang/MM2024/model_dir/random/noise/Mixup/2024-03-09_16-25-17/model.pkl'
mixup_path='/home/jcjiang/MM2024/model_dir/random/Mixup/2024-03-09_16-24-56/model.pkl'

# i=0
# ### Common hyperparameters
# * max_epoch:120
# * lr: 0.001 or 0.005

# ### Hyperparameters for PACS(ResNet-18) (Corresponding to each task in order, ACPS)

# | Method | A | C | P | S |
# |----------|----------|----------|----------|----------|
# |DANN(alpha)|0.5| 1| 0.1| 0.1|
# |Mixup(mixupalpha)|0.1| 0.2| 0.2| 0.2|
# |RSC(rsc_f_drop_factor,rsc_b_drop_factor)|0.1,0.3|0.3,0.1|0.1,0.1|0.1,0.1|
# |MMD(mmd_gamma)|10|1|0.5|0.5|
# |CORAL(mmd_gamma)|0.5|0.1|1|0.01|
# |GroupDRO(groudro_eta)|10**(-2.5)| 0.001| 0.001| 0.01|
# |ANDMask(tau)|0.5|0.82|0.5|0.5|
# |VREx(lam,anneal_iters)|1,5000|1,100|0.3,5000|1,10|

# # MLDG 
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/MLDG \
# --test_envs $test_envs --dataset $dataset --algorithm ${algorithm[i]} --mldg_beta 10


# # ERM
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
# --task img_dg --output $output --test_envs 1 --dataset PACS --algorithm ERM --lr 0.005  --use_pretrained --gpu_id 0 &
# # DANN
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task img_dg --output $output \
# --test_envs 1 --dataset PACS --algorithm DANN --alpha 1 --lr 0.005 --use_pretrained --gpu_id 0 --use_qat &
# RSC
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task img_dg --output $output \
# --test_envs 1 --dataset PACS --algorithm RSC --rsc_f_drop_factor 0.1 --rsc_b_drop_factor 0.1 --lr 0.005 --use_pretrained --gpu_id 0 &
# # # CORAL
# # python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task img_dg --output $output \
# # --test_envs 1 --dataset PACS --algorithm CORAL --mmd_gamma 0.1 --lr 0.005 --use_pretrained --gpu_id 0 
# # Mixup
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task img_dg --output $output \
# --test_envs 1 --dataset PACS --algorithm Mixup --mixupalpha 0.2 --lr 0.005  --use_pretrained --gpu_id 0 &
# # MMD
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task img_dg --output $output \
# --test_envs 1 --dataset PACS --algorithm MMD --mmd_gamma 1 --lr 0.005 --use_pretrained --gpu_id 0 &
# # Group_DRO
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task img_dg --output  $output/Group_DRO \
# --test_envs 1 --dataset PACS --algorithm GroupDRO --groupdro_eta 0.001 --lr 0.005   --gpu_id 0 --use_pretrained &

# # ANDMask
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/ANDMask \
# --test_envs 1 --dataset $dataset --algorithm ANDMask --tau 0.82 --lr 0.005  --gpu_id 0 --use_pretrained &

# # VREx
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/VREx \
# --test_envs 1 --dataset $dataset --algorithm VREx --lam 1 --anneal_iters 100 --lr 0.005 --gpu_id 0 --use_pretrained &


# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task img_dg --output $output \
# --test_envs 1 --dataset PACS --algorithm RSC --rsc_f_drop_factor 0.3 --rsc_b_drop_factor 0.1 --lr 0.005 --use_pretrained --use_qat --gpu_id 0 


# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task img_dg --output $output \
# --test_envs 1 --dataset PACS --algorithm RSC --rsc_f_drop_factor 0.3 --rsc_b_drop_factor 0.1 --lr 0.005 --use_pretrained --use_qat --gpu_id 0


# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task img_dg --output $output \
# --test_envs 1 --dataset PACS --algorithm MMD --mmd_gamma 1 --lr 0.005 --use_pretrained --use_qat 



# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task img_dg --output $output \
# --test_envs 1 --dataset PACS --algorithm CORAL --mmd_gamma 0.1 --lr 0.005 --use_pretrained --use_qat

# # wait




# wait


# DANN
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task img_dg --output $output/DANN_QAT \
# --test_envs 1 --dataset PACS --algorithm DANN --alpha 1 --lr 0.005  --use_qat --gpu_id 0 &
# # Mixup
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task img_dg --output $output/Mixup_QAT \
# --test_envs 1 --dataset PACS --algorithm Mixup --mixupalpha 0.2 --lr 0.005  --gpu_id 0 &
# # RSC
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task img_dg --output $output/RSC_QAT \
# --test_envs 1 --dataset PACS --algorithm RSC --rsc_f_drop_factor 0.3 --rsc_b_drop_factor 0.1 --lr 0.005 --use_qat --gpu_id 0 &
# # MMD
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task img_dg --output $output/MMD_QAT \
# --test_envs 1 --dataset PACS --algorithm MMD --mmd_gamma 1 --lr 0.005  --gpu_id 0 &
# # CORAL
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task img_dg --output $output/CORAL_QAT \
# --test_envs 1 --dataset PACS --algorithm CORAL --mmd_gamma 0.1 --lr 0.005 --use_qat --gpu_id 0 &

# # Group_DRO
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task img_dg --output $output/Group_DRO_QAT \
# --test_envs 1 --dataset PACS --algorithm GroupDRO --groupdro_eta 0.001 --lr 0.005 --gpu_id 0 --use_qat &

# # ANDMask
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/ANDMask_QAT \
# --test_envs 1 --dataset $dataset --algorithm ANDMask --tau 0.82 --lr 0.005  --gpu_id 0 --use_qat &

# # VREx
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/VREx_QAT \
# --test_envs 1 --dataset $dataset --algorithm VREx --lam 1 --anneal_iters 100 --lr 0.005 --gpu_id 0 --use_qat &

# wait


# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/ANDMask \
# --test_envs 1 --dataset $dataset --algorithm ANDMask --tau 0.5 --lr 0.005 --use_qat



# # The following experiments are running on the singularity cluster of MSRA.The environment are shown in the following file.
# # CUDA version, wget https://dgresearchredmond.blob.core.windows.net/amulet/projects/lw/singularity/env/env1.txt
# # GPU information, wget https://dgresearchredmond.blob.core.windows.net/amulet/projects/lw/singularity/env/env2.txt
# # python package information, wget https://dgresearchredmond.blob.core.windows.net/amulet/projects/lw/singularity/env/env.txt
# python train.py --data_dir $data_dir --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output /home/lw/lw/data/train_output/difexpacs/2-0 \
# --test_envs 0 --dataset PACS --algorithm DIFEX --alpha 0.1 --beta 0.01 --lam 0.1 --disttype 2-norm
# python train.py --data_dir $data_dir --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output /home/lw/lw/data/train_output/difexpacs/2-1 \
# --test_envs 1 --dataset PACS --algorithm DIFEX --alpha 0.001 --beta 0 --lam 0.01 --disttype 2-norm
# python train.py --data_dir $data_dir --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output /home/lw/lw/data/train_output/difexpacs/2-2 \
# --test_envs 2 --dataset PACS --algorithm DIFEX --alpha 0.001 --beta 0.01 --lam 0.01 --disttype 2-norm
# python train.py --data_dir $data_dir --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output /home/lw/lw/data/train_output/difexpacs/2-3 \
# --test_envs 3 --dataset PACS --algorithm DIFEX --alpha 0.01 --beta 1 --lam 0 --disttype 2-norm

# python train.py --data_dir $data_dir --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output /home/lw/lw/data/train_output/difexpacs/n1n-0 \
# --test_envs 0 --dataset PACS --algorithm DIFEX --alpha 0.1 --beta 0.1 --lam 0 --disttype norm-1-norm
# python train.py --data_dir $data_dir --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output /home/lw/lw/data/train_output/difexpacs/n1n-1 \
# --test_envs 1 --dataset PACS --algorithm DIFEX --alpha 0.001 --beta 1 --lam 0.01 --disttype norm-1-norm
# python train.py --data_dir $data_dir --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output /home/lw/lw/data/train_output/difexpacs/n1n-2 \
# --test_envs 2 --dataset PACS --algorithm DIFEX --alpha 0.001 --beta 0.5 --lam 0.1 --disttype norm-1-norm
# python train.py --data_dir $data_dir --max_epoch 120 --net resnet18 --checkpoint_freq 1 --task img_dg --output /home/lw/lw/data/train_output/difexpacs/n1n-3 \
# --test_envs 3 --dataset PACS --algorithm DIFEX --alpha 0.01 --beta 10 --lam 1 --disttype norm-1-norm


# # RSC_GT
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task img_dg --output $output \
# --test_envs 1 --dataset PACS --algorithm RSC_GT --rsc_f_drop_factor 0.1 --rsc_b_drop_factor 0.1 --lr 0.005 --use_pretrained --gpu_id 0  &

# # RSC_RANDOM
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task img_dg --output $output \
# --test_envs 1 --dataset PACS --algorithm RSC_RANDOM --rsc_f_drop_factor 0.1 --rsc_b_drop_factor 0.1 --lr 0.005 --use_pretrained --gpu_id 0  &

# # RSC
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task img_dg --output $output \
# --test_envs 1 --dataset PACS --algorithm RSC --rsc_f_drop_factor 0.1 --rsc_b_drop_factor 0.1 --lr 0.005 --use_pretrained --use_qat --gpu_id 0  &



# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task img_dg --output $output \
# --test_envs 1 --dataset PACS --algorithm RSC --rsc_f_drop_factor 0.05 --rsc_b_drop_factor 0.1 --lr 0.005 --use_pretrained --gpu_id 0  --use_qat &



# # RSC ERM
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output --test_envs 1 --dataset PACS --algorithm ERM --lr 0.005  \
#     --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config4.yaml\
#     --use_pretrained --gpu_id 0 --use_qat \
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/RSC/2023-12-18_11-23-56/model.pkl 


# # ERM ERM
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output --test_envs 1 --dataset PACS --algorithm ERM --lr 0.005  \
#     --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config4.yaml \
#     --use_pretrained --gpu_id 0 --use_qat \
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/ERM/2023-12-19_11-02-42/model.pkl

# # MMD ERM
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output --test_envs 1 --dataset PACS --algorithm ERM --lr 0.005  \
#     --use_pretrained --gpu_id 0 --use_qat \
#     --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config4.yaml \
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/MMD/2023-12-18_11-23-56/model.pkl

# # MIXUP ERM
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output --test_envs 1 --dataset PACS --algorithm ERM --lr 0.005  \
#     --use_pretrained --gpu_id 0 --use_qat \
#     --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config4.yaml \
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/Mixup/2023-12-18_11-23-56/model.pkl

# # ANDMASK ERM
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output --test_envs 1 --dataset PACS --algorithm ERM --lr 0.005  \
#     --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config4.yaml \
#     --use_pretrained --gpu_id 0 --use_qat \
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/ANDMask/2023-12-19_11-10-08/model.pkl


# # RSC ERM
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output --test_envs 1 --dataset PACS --algorithm ERM --lr 0.005  \
#     --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config2.yaml\
#     --use_pretrained --gpu_id 0 --use_qat \
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/RSC/2023-12-18_11-23-56/model.pkl 


# # ERM ERM
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output --test_envs 1 --dataset PACS --algorithm ERM --lr 0.005  \
#     --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config2.yaml\
#     --use_pretrained --gpu_id 0 --use_qat \
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/ERM/2023-12-19_11-02-42/model.pkl 

# # MMD ERM
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output --test_envs 1 --dataset PACS --algorithm ERM --lr 0.005  \
#     --use_pretrained --gpu_id 0 --use_qat \
#     --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config2.yaml\
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/MMD/2023-12-18_11-23-56/model.pkl 

# # MIXUP ERM
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output --test_envs 1 --dataset PACS --algorithm ERM --lr 0.005  \
#     --use_pretrained --gpu_id 0 --use_qat \
#     --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config2.yaml\
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/Mixup/2023-12-18_11-23-56/model.pkl

# # ANDMASK ERM
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output --test_envs 1 --dataset PACS --algorithm ERM --lr 0.005  \
#     --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config2.yaml\
#     --use_pretrained --gpu_id 0 --use_qat \
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/Combined_ANDMASK_MMD/Combined_ANDMASK_MMD/2024-02-06_20-43-00/model.pkl


# # MMD initial ANDMask
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output/MMD_ininal_ANDMask --test_envs 1 --dataset PACS --algorithm ERM --lr 0.005  \
#     --use_pretrained --gpu_id 0  \
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/MMD/2023-12-18_11-23-56/model.pkl
#     # --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/ANDMask/2023-12-19_11-10-08/model.pkl
    

# ERM ANDMask
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/ANDMask_ERM \
#     --test_envs 1 --dataset $dataset --algorithm ANDMask --tau 0.82 --lr 0.005  --gpu_id 0 --use_qat --use_pretrained \
#     --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config2.yaml\
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/ERM/2023-12-19_11-02-42/model.pkl &
# # RSC ANDMask
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/ANDMask_RSC \
#     --test_envs 1 --dataset $dataset --algorithm ANDMask --tau 0.82 --lr 0.005  --gpu_id 0 --use_qat --use_pretrained \
#     --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config2.yaml\
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/RSC/2023-12-18_11-23-56/model.pkl &

# # ANDMASK ANDMask
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/ANDMask_ANDMask \
#     --test_envs 1 --dataset $dataset --algorithm ANDMask --tau 0.82 --lr 0.005  --gpu_id 0 --use_qat --use_pretrained \
#     --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config2.yaml\
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/ANDMask/2023-12-19_11-10-08/model.pkl &





# # Group_DRO
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output  $output --test_envs 1 --dataset PACS --algorithm GroupDRO --groupdro_eta 0.001 --lr 0.005 \
#     --gpu_id 0 --use_pretrained  

# # CORAL
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output --test_envs 1 --dataset PACS --algorithm CORAL --mmd_gamma 0.1 --lr 0.005 \
#     --use_pretrained --gpu_id 0 

# # Group_DRO
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output --test_envs 1 --dataset PACS --algorithm ERM --lr 0.005  \
#     --use_pretrained --gpu_id 0 --use_qat \
#     --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config2.yaml\
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/Mixup/2023-12-18_11-23-56/model.pkl




# # ERM combined MMD & ANDMask
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/Combined_ANDMASK_MMD \
#     --test_envs 1 --dataset $dataset --algorithm Combined_ANDMASK_MMD --tau 0.82 --lr 0.005  --mmd_gamma 1 --gpu_id 0  --use_pretrained \
#     --use_qat --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config2.yaml --mmd_average \
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/ERM/2023-12-19_11-02-42/model.pkl &

# # RSC combined MMD & ANDMask
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/Combined_ANDMASK_MMD \
#     --test_envs 1 --dataset $dataset --algorithm Combined_ANDMASK_MMD --tau 0.82 --lr 0.005  --mmd_gamma 1 --gpu_id 0  --use_pretrained \
#     --use_qat --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config2.yaml --mmd_average \
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/RSC/2023-12-18_11-23-56/model.pkl &

# # ANDMASK combined MMD & ANDMask
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/Combined_ANDMASK_MMD \
#     --test_envs 1 --dataset $dataset --algorithm Combined_ANDMASK_MMD --tau 0.82 --lr 0.005  --mmd_gamma 1 --gpu_id 0  --use_pretrained \
#     --use_qat --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config2.yaml --mmd_average \
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/ANDMask/2023-12-19_11-10-08/model.pkl &

# # MMD combined MMD & ANDMask
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/Combined_ANDMASK_MMD \
#     --test_envs 1 --dataset $dataset --algorithm Combined_ANDMASK_MMD --tau 0.82 --lr 0.005  --mmd_gamma 1 --gpu_id 0  --use_pretrained \
#     --use_qat --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config2.yaml --mmd_average \
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/MMD/2023-12-18_11-23-56/model.pkl &

# # MIXUP combined MMD & ANDMask
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/MIXUP_Combined_ANDMASK_MMD \
#     --test_envs 1 --dataset $dataset --algorithm Combined_ANDMASK_MMD --tau 0.82 --lr 0.005  --mmd_gamma 1 --gpu_id 0  --use_pretrained \
#     --use_qat --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config2.yaml --mmd_average \
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/Mixup/2023-12-18_11-23-56/model.pkl &







# # RSC ANDMask
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/ANDMask_RSC_0.7_disturb105 \
#     --test_envs 1 --dataset $dataset --algorithm ANDMask --tau 0.7 --lr 0.005  --gpu_id 0 --use_qat --use_pretrained --use_disturb \
#     --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config2.yaml\
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/RSC/2023-12-18_11-23-56/model.pkl &


# # MMD ANDMask
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/ANDMask_MMD_0.7_disturb105 \
#     --test_envs 1 --dataset $dataset --algorithm ANDMask --tau 0.7 --lr 0.005  --gpu_id 0 --use_qat --use_pretrained --use_disturb \
#     --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config2.yaml\
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/MMD/2023-12-18_11-23-56/model.pkl &


# # MIXUP ANDMask
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/ANDMask_MIXUP_0.7_disturb105 \
#     --test_envs 1 --dataset $dataset --algorithm ANDMask --tau 0.7 --lr 0.005  --gpu_id 0 --use_qat --use_pretrained --use_disturb \
#     --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config2.yaml\
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/Mixup/2023-12-18_11-23-56/model.pkl &

# # MIXUP ANDMask
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/ANDMask_MIXUP_0.7_disturb120_layer3conv \
#     --test_envs 1 --dataset $dataset --algorithm ANDMask --tau 0.7 --lr 0.005  --gpu_id 0 --use_qat --use_pretrained --use_disturb --perturbation_ratio 0.2\
#     --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config2.yaml\
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/Mixup/2023-12-18_11-23-56/model.pkl &

# # MIXUP ANDMask
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/ANDMask_MIXUP_0.7_disturb120_conv \
#     --test_envs 1 --dataset $dataset --algorithm ANDMask --tau 0.7 --lr 0.005  --gpu_id 0 --use_qat --use_pretrained --use_disturb --perturbation_ratio 0.2\
#     --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config2.yaml\
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/Mixup/2023-12-18_11-23-56/model.pkl &
# # MIXUP ANDMask
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/ANDMask_MIXUP_0.7_disturb140_layer3_conv \
#     --test_envs 1 --dataset $dataset --algorithm ANDMask --tau 0.7 --lr 0.005  --gpu_id 0 --use_qat --use_pretrained --use_disturb --perturbation_ratio 0.4\
#     --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config2.yaml\
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/Mixup/2023-12-18_11-23-56/model.pkl &

# # MIXUP ANDMask
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/ANDMask_MIXUP_0.3_disturb105 \
#     --test_envs 1 --dataset $dataset --algorithm ANDMask --tau 0.3 --lr 0.005  --gpu_id 0 --use_qat --use_pretrained --use_disturb \
#     --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config2.yaml\
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/Mixup/2023-12-18_11-23-56/model.pkl &


# # MIXUP ANDMask
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/ANDMask_MIXUP_0.3_disturb105 \
#     --test_envs 1 --dataset $dataset --algorithm ANDMask --tau 0.3 --lr 0.005  --gpu_id 0 --use_qat --use_pretrained --use_disturb \
#     --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config2.yaml\
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/Mixup/2023-12-18_11-23-56/model.pkl &

# # MIXUP ANDMask
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/ANDMask_MIXUP_randn0.02_layer4conv \
#     --test_envs 1 --dataset $dataset --algorithm ANDMask --tau 0.7 --lr 0.005  --gpu_id 0 --use_qat --use_pretrained\
#     --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config2.yaml\
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/Mixup/2023-12-18_11-23-56/model.pkl &

# # MIXUP ANDMask
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/ANDMask_MIXUP_randn0.02_layer4conv \
#     --test_envs 1 --dataset $dataset --algorithm ANDMask --tau 0.3 --lr 0.005  --gpu_id 0 --use_qat --use_pretrained\
#     --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config2.yaml\
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/Mixup/2023-12-18_11-23-56/model.pkl &


# # random ANDMask
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/ANDMask_random \
#     --test_envs 1 --dataset $dataset --algorithm ANDMask --tau 0.7 --lr 0.005  --gpu_id 0 --use_qat\
#     --config_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/config2.yaml\
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/Mixup/2023-12-18_11-23-56/model.pkl &


# # ANDMASK ANDMask
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/ANDMask_ANDMASK_cluster\
#     --test_envs 1 --dataset $dataset --algorithm ERM --tau 0.7 --lr 0.005  --gpu_id 0  --use_pretrained \
#     --use_qat --config_path /home/jcjiang/MM2024/config2.yaml \
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/ANDMask/2023-12-19_11-10-08/model.pkl &


# # RSC ANDMask
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/ANDMask_RSC_0.7_disturb101 \
#     --test_envs 1 --dataset $dataset --algorithm ANDMask --tau 0.7 --lr 0.005  --gpu_id 0 --use_qat --use_pretrained --use_disturb \
#     --config_path /home/jcjiang/MM2024/config2.yaml\
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/RSC/2023-12-18_11-23-56/model.pkl &


# # MMD ANDMask
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/ANDMask_MMD_0.7_disturb101 \
#     --test_envs 1 --dataset $dataset --algorithm ANDMask --tau 0.7 --lr 0.005  --gpu_id 0 --use_qat --use_pretrained --use_disturb \
#     --config_path /home/jcjiang/MM2024/config2.yaml\
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/MMD/2023-12-18_11-23-56/model.pkl &


# # MIXUP ANDMask
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/ANDMask_MIXUP_0.7_disturb101 \
#     --test_envs 1 --dataset $dataset --algorithm ANDMask --tau 0.7 --lr 0.005  --gpu_id 0 --use_qat --use_pretrained --use_disturb \
#     --config_path /home/jcjiang/MM2024/config2.yaml\
#     --local_model_path /home/jcjiang/first_year/research/OOD+Quantization/pre_test/DeepDG/out/Mixup/2023-12-18_11-23-56/model.pkl &

# # ERM ERM
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output --test_envs 1 --dataset PACS --algorithm ERM --lr 0.005  \
#     --config_path /home/jcjiang/MM2024/config2.yaml\
#     --use_pretrained --gpu_id 0 --use_qat \
#     --local_model_path /home/jcjiang/MM2024/model_dir/ERM/2023-12-19_11-02-42/model.pkl &

# RSC ERM
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output --test_envs 1 --dataset PACS --algorithm ERM --lr 0.005  \
#     --config_path /home/jcjiang/MM2024/config2.yaml\
#     --use_pretrained --gpu_id 0 --use_qat \
#     --local_model_path /home/jcjiang/MM2024/model_dir/RSC/2023-12-18_11-23-56/model.pkl &
# # ANDMASK ERM
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output --test_envs 1 --dataset PACS --algorithm ERM --lr 0.005  \
#     --config_path /home/jcjiang/MM2024/config2.yaml\
#     --use_pretrained --gpu_id 0 --use_qat \
#     --local_model_path /home/jcjiang/MM2024/model_dir/ANDMASK/2023-12-19_11-10-08/model.pkl &


# # MMD ERM
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output --test_envs 1 --dataset PACS --algorithm ERM --lr 0.005  \
#     --use_pretrained --gpu_id 0 --use_qat \
#     --config_path /home/jcjiang/MM2024/config2.yaml\
#     --local_model_path /home/jcjiang/MM2024/model_dir/MMD/2023-12-18_11-23-56/model.pkl &

# # MIXUP ERM
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output --test_envs 1 --dataset PACS --algorithm ERM --lr 0.005  \
#     --use_pretrained --gpu_id 0 --use_qat \
#     --config_path /home/jcjiang/MM2024/config2.yaml\
#     --local_model_path /home/jcjiang/MM2024/model_dir/Mixup/2023-12-18_11-23-56/model.pkl &


# # img pretrained QAT
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net --task $task --output $output/img_preatrained_QAT \
#     --test_envs 1 --dataset $dataset --algorithm ERM --tau 0.82 --lr 0.005  --mmd_gamma 1 --gpu_id 0  --use_pretrained \
#     --use_qat --config_path /home/jcjiang/MM2024/config2.yaml \


# # MIXUP 
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output/32bit_QAT --test_envs 1 --dataset PACS --algorithm Mixup --lr 0.005  \
#     --local_model_path /home/jcjiang/MM2024/model_dir/Mixup/2023-12-18_11-23-56/model.pkl \
#     --gpu_id 0  --mixupalpha 0.2 --config_path /home/jcjiang/MM2024/config2.yaml &
    


# # ERM 
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output/32bit_QAT_ERM  --test_envs 1 --dataset PACS --algorithm ERM --lr 0.005  \
#     --gpu_id 0 --use_qat --config_path /home/jcjiang/MM2024/config2.yaml \
#     --local_model_path /home/jcjiang/MM2024/model_dir/random/ERM/2024-03-09_16-24-56/model.pkl &


# # ANDMASK
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output/32bit_QAT_ANDMask  --test_envs 1 --dataset PACS --algorithm ERM --lr 0.005  \
#     --gpu_id 0 --use_qat --tau 0.7 --config_path /home/jcjiang/MM2024/config2.yaml \
#     --local_model_path /home/jcjiang/MM2024/model_dir/random/ANDMask/2024-03-09_16-24-56/model.pkl &


# # Mixup ANDMask
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output/32bit_ANDMaskQAT_Mixup  --test_envs 1 --dataset PACS --algorithm ANDMask --lr 0.005  \
#     --gpu_id 0 --use_qat --tau 0.3 --config_path /home/jcjiang/MM2024/config2.yaml \
#     --local_model_path /home/jcjiang/MM2024/model_dir/random/Mixup/2024-03-09_16-24-56/model.pkl &


# # # Mixup ANDMask
# # python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
# #     --task img_dg --output $output/32bit_ANDMaskQAT_ANDMASK  --test_envs 1 --dataset PACS --algorithm ANDMask --lr 0.005  \
# #     --gpu_id 0 --use_qat --tau 0.7 --config_path /home/jcjiang/MM2024/config2.yaml \
# #     --local_model_path /home/jcjiang/MM2024/model_dir/random/Mixup/2024-03-09_16-24-56/model.pkl &

# # noiseMIXUP noise
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output/32bitnoise_initial_add_noise_QAT_Mixup --test_envs 1 --dataset PACS --algorithm ERM --lr 0.005  \
#     --use_qat  --config_path /home/jcjiang/MM2024/config2.yaml \
#     --gpu_id 0  --mixupalpha 0.2 --use_disturb --perturbation_ratio 0.02 &


# # ERMnoise ANDMaskQAT
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output/32bitnoise/ANDMaskQAT/ERM  --test_envs 1 --dataset PACS --algorithm ANDMask --lr 0.005  \
#     --gpu_id 0 --use_qat --tau 0.7 --config_path /home/jcjiang/MM2024/config2.yaml \
#     --local_model_path $erm_noise_path &


# # Mixup noise initial noise QAT
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output/32bitnoise/initial_noise/QAT/Mixup  --test_envs 1 --dataset PACS --algorithm ERM --lr 0.005  \
#     --gpu_id 0 --use_qat --tau 0.7 --config_path /home/jcjiang/MM2024/config2.yaml \
#     --local_model_path $mixup_noise_path --use_disturb --perturbation_ratio 0.02&



# # ERM ANDMask QAT
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output/32bit/ANDMaskQAT/ERM  --test_envs 1 --dataset PACS --algorithm ERM --lr 0.005  \
#     --gpu_id 0 --use_qat --tau 0.3 --config_path /home/jcjiang/MM2024/config3.yaml \
#     --local_model_path $erm_path&

# # ERM ANDMaskQAT
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output/32bit/ANDMaskQAT/ERM  --test_envs 1 --dataset PACS --algorithm ANDMask --lr 0.005  \
#     --gpu_id 0 --use_qat --tau 0.7 --config_path /home/jcjiang/MM2024/config4.yaml \
#     --local_model_path $erm_path&



# # ANDMaskQAT
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output/32bit/initial_noise/ANDMaskQAT/Mixup  --test_envs 1 --dataset PACS --algorithm ANDMask --lr 0.005  \
#     --gpu_id 0 --use_qat --tau 0.3 --config_path /home/jcjiang/MM2024/config2.yaml \
#     --local_model_path $mixup_path --use_disturb --perturbation_ratio 0.02&

# # ANDMaskQAT
# python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
#     --task img_dg --output $output/32bit/initial_noise/ANDMaskQAT/ERM  --test_envs 1 --dataset PACS --algorithm ANDMask --lr 0.005  \
#     --gpu_id 0 --use_qat --tau 0.3 --config_path /home/jcjiang/MM2024/config2.yaml \
#     --local_model_path $erm_path --use_disturb --perturbation_ratio 0.02&


# 32bitnoise noise initial ANDMaskQAT
python train.py --data_dir $data_dir --max_epoch $max_epoch --net $net \
    --task img_dg --output $output/32bitnoise/initial_noise/ANDMaskQAT/ANDMask  --test_envs 1 --dataset PACS --algorithm ANDMask --lr 0.005  \
    --gpu_id 0 --use_qat --tau 0.7 --config_path /home/jcjiang/MM2024/config2.yaml \
    --local_model_path $andmask_noise_path --use_disturb --perturbation_ratio 0.02&
