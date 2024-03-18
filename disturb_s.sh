# ===========================================================
# 2d loss contours for ResNet-56
# ===========================================================
dataset='PACS' # office-home of PACS
algorithm=('MLDG' 'ERM' 'DANN' 'RSC' 'Mixup' 'MMD' 'CORAL' 'VREx') 
test_envs=1
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

# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/1.preatrained_noise_random_qat/preatrained_noise_erm/erm_6246.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --use_qat --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS  --config_path /home/jcjiang/loss-landscape/config2.yaml &

# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/1.preatrained_noise_random_qat/random_ERM_bit2/2024-03-09_19-01-56/model.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --use_qat --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS  --config_path /home/jcjiang/loss-landscape/config2.yaml &

# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/2.andmask/andmask_erm/model.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --use_qat --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS  --config_path /home/jcjiang/loss-landscape/config2.yaml &

# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/2.andmask/erm/model.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --use_qat --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS  --config_path /home/jcjiang/loss-landscape/config2.yaml &


# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/4.random_vs_preatrain/1/ERM/2023-12-19_11-02-42/model.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter  --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS  &

# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/4.random_vs_preatrain/1/ERM_copy/2024-03-09_16-24-56/model.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter  --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS  &

# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/6.tau/0.3/2024-03-10_12-26-03/model.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --use_qat --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS  --config_path /home/jcjiang/loss-landscape/config2.yaml &

# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/6.tau/0.7/2024-03-10_11-05-24/model.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --use_qat --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS  --config_path /home/jcjiang/loss-landscape/config2.yaml &

# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/3.andmask_mixup32/ANDMASK/andmask/model.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter  --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS   &

# # good1
# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/good/andmask_qat64/model.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --use_qat --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS  --config_path /home/jcjiang/loss-landscape/config2.yaml --val&
# # good2
# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/good/andmasktau0.3_0.66/model.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --use_qat --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS  --config_path /home/jcjiang/loss-landscape/config2.yaml --val&


# # 4.3.1
# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/4.random_vs_preatrain/3/andmask_erm/model.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --use_qat --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS  --config_path /home/jcjiang/loss-landscape/config2.yaml --val&
# # 4.3.2
# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/4.random_vs_preatrain/3/erm_random_andmaskqat/2024-03-09_23-50-44/model.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --use_qat --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS  --config_path /home/jcjiang/loss-landscape/config2.yaml --val&

# # 4.3.2
# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/4.random_vs_preatrain/3/erm_random_andmaskqat/2024-03-09_23-50-44/model.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --use_qat --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS  --config_path /home/jcjiang/loss-landscape/config2.yaml --val&

# # 4.3.2
# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/4.random_vs_preatrain/3/erm_random_andmaskqat/2024-03-09_23-50-44/model.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --use_qat --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS  --config_path /home/jcjiang/loss-landscape/config2.yaml --val&



# # 4.3.2
# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/4.random_vs_preatrain/3/erm_random_andmaskqat/2024-03-09_23-50-44/model.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --use_qat --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS  --config_path /home/jcjiang/loss-landscape/config2.yaml --val&

# # 4.3.2
# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/4.random_vs_preatrain/3/erm_random_andmaskqat/2024-03-09_23-50-44/model.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --use_qat --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS  --config_path /home/jcjiang/loss-landscape/config2.yaml --val&



# # 1.1
# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/4.random_vs_preatrain/3/erm_random_andmaskqat/2024-03-09_23-50-44/model.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --use_qat --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS  --config_path /home/jcjiang/loss-landscape/config2.yaml --val&

# # 1.2
# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/4.random_vs_preatrain/3/erm_random_andmaskqat/2024-03-09_23-50-44/model.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --use_qat --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS  --config_path /home/jcjiang/loss-landscape/config2.yaml --val&



# # 2.1
# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/4.random_vs_preatrain/3/erm_random_andmaskqat/2024-03-09_23-50-44/model.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --use_qat --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS  --config_path /home/jcjiang/loss-landscape/config2.yaml --val&

# # 2.2
# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/4.random_vs_preatrain/3/erm_random_andmaskqat/2024-03-09_23-50-44/model.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --use_qat --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS  --config_path /home/jcjiang/loss-landscape/config2.yaml --val&


# # 3.1
# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/4.random_vs_preatrain/3/erm_random_andmaskqat/2024-03-09_23-50-44/model.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --use_qat --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS  --config_path /home/jcjiang/loss-landscape/config2.yaml --val&

# # 3.2
# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/4.random_vs_preatrain/3/erm_random_andmaskqat/2024-03-09_23-50-44/model.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --use_qat --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS  --config_path /home/jcjiang/loss-landscape/config2.yaml --val&


# # 4.3.1
# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/4.random_vs_preatrain/3/erm_random_andmaskqat/2024-03-09_23-50-44/model.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --use_qat --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS  --config_path /home/jcjiang/loss-landscape/config2.yaml --val&

# # 4.3.2
# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/4.random_vs_preatrain/3/erm_random_andmaskqat/2024-03-09_23-50-44/model.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --use_qat --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS  --config_path /home/jcjiang/loss-landscape/config2.yaml --val&


# # 5.1
# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/5.noise_vs_onoise/erm_random_andmaskqat/2024-03-09_23-50-44/model.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --use_qat --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS  --config_path /home/jcjiang/loss-landscape/config2.yaml --val&

# # 5.2
# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/5.noise_vs_onoise/erm_random_noise_andmaskqat/2024-03-10_16-08-01/model.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter --use_qat --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS  --config_path /home/jcjiang/loss-landscape/config2.yaml --val&


# # 4.2.2 val
# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/4.random_vs_preatrain/2/ANDMask/2024-03-09_16-24-56/model.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter  --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS  --val&

# # 4.2.2 target
# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/4.random_vs_preatrain/2/ANDMask/2024-03-09_16-24-56/model.pkl\
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter  --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS   &



# # 1.1 val
# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/1.preatrained_noise_random_qat/preatrained_noise_erm/erm_6246.pkl \
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter  --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS --use_qat --config_path /home/jcjiang/loss-landscape/config2.yaml --val&

# # 1.2 val
# mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
#     --model_file /home/jcjiang/loss-landscape/compare_model/1.preatrained_noise_random_qat/random_ERM_bit2/2024-03-09_19-01-56/model.pkl\
#     --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter  --data_dir $data_dir --net $net \
#     --task img_dg  --test_envs 1 --dataset PACS   --use_qat --config_path /home/jcjiang/loss-landscape/config2.yaml --val&

# 4.3.1 val
mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
    --model_file /home/jcjiang/loss-landscape/compare_model/4.random_vs_preatrain/3/andmask_erm/model.pkl \
    --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter  --data_dir $data_dir --net $net \
    --task img_dg  --test_envs 1 --dataset PACS --use_qat --config_path /home/jcjiang/loss-landscape/config2.yaml --val --disturb_s&

# 4.3.1 target
mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
    --model_file /home/jcjiang/loss-landscape/compare_model/4.random_vs_preatrain/3/andmask_erm/model.pkl\
    --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter  --data_dir $data_dir --net $net \
    --task img_dg  --test_envs 1 --dataset PACS   --use_qat --config_path /home/jcjiang/loss-landscape/config2.yaml  --disturb_s&

# 4.3.2 val
mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
    --model_file /home/jcjiang/loss-landscape/compare_model/4.random_vs_preatrain/3/erm_random_andmaskqat/2024-03-09_23-50-44/model.pkl \
    --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter  --data_dir $data_dir --net $net \
    --task img_dg  --test_envs 1 --dataset PACS --use_qat --config_path /home/jcjiang/loss-landscape/config2.yaml --val --disturb_s&

# 4.3.2 target
mpirun -n 1 python plot_surface.py --x=-1:1:51 --y=-1:1:51  \
    --model_file /home/jcjiang/loss-landscape/compare_model/4.random_vs_preatrain/3/erm_random_andmaskqat/2024-03-09_23-50-44/model.pkl\
    --mpi --cuda --dir_type weights --xignore biasbn --xnorm filter --yignore biasbn --ynorm filter  --data_dir $data_dir --net $net \
    --task img_dg  --test_envs 1 --dataset PACS   --use_qat --config_path /home/jcjiang/loss-landscape/config2.yaml  --disturb_s&


