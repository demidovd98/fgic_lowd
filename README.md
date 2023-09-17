# fgic_lowd


cd /l/users/20020067/Activities/FGIC/low_data/Combined/Refine/FFVT_my_combined

module load cuda-10.2
nvcc --version

source /apps/local/anaconda2023/conda_init.sh 
conda activate ffvt_u2n_p36

python3 -W ignore -m torch.distributed.launch --nproc_per_node 1 train.py --name air_ft_05LossFocal01_resnet50vanilla_1k_224_bs24_doubleCrop2564n6481_sepNfix_autoSched05_LrRatio10_rn50_40k_noSAMtrashVanilla_doubleAugs224_KLlossSAM001_batchmean_inputLog_lr003_ld10 --dataset air --model_type ViT-B_16 --img_size 224 --resize_size 256 --train_batch_size 24 --eval_batch_size 24 --learning_rate 0.03 --num_steps 40000 --fp16 --eval_every 200 --pretrained_dir checkpoint/ViT-B_16.npz --split 10 --lr_ratio 10.0 --dist_coef 0.01

TEST  PR
