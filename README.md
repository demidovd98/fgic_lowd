# fgic_lowd



### Instructions:

#### Run:

1. Initialise the code space:

- Public:
```
cd '/path/to/cloned/repo/'

bash init.sh 

conda activate ffvt_u2n_p36
```

- Dev:
```
cd '/path/to/cloned/repo/'

bash init_dev.sh

source /apps/local/anaconda2023/conda_init.sh
conda activate /l/users/cv-805/envs/ffvt_u2n_p36
```


2. Run (example):
```
python3 -W ignore -m torch.distributed.launch --nproc_per_node 1 train.py --name CUB_ft_05LossFocal01_resnet50vanilla_1k_224_bs24_doubleCrop2564n6481_sepNfix_autoSched05_LrRatio10_rn50_40k_noSAMtrashVanilla_doubleAugs224_KLlossSAM001_batchmean_inputLog_lr003_ld10_test --dataset CUB --model_type cnn --model_name resnet50 --img_size 224 --resize_size 256 --train_batch_size 24 --eval_batch_size 24 --learning_rate 0.03 --num_steps 40000 --fp16 --eval_every 200 --pretrained_dir checkpoint/ViT-B_16.npz --split 10 --lr_ratio 10.0 --dist_coef 0.01 --aug_type double_crop
```


### Other:

#### Create environment:

pip:

```
python3 -m venv env
source env/bin/activate
pip install -r requirements_pip.txt
```

conda:

1. With environment.yml:

```
conda env create -f environment.yml
```

2. With requirements.txt:

```
conda create --name <env> --file requirements_conda.txt
```


#### Copy environment to the shared folder:

```
conda create --prefix ./name --clone name
```


#### Use local environment:
```
cd '/l/users/20020067/Activities/FGIC/low_data/Combined/Refine/fgic_lowd'

module load cuda-10.2
nvcc --version

source /apps/local/anaconda2023/conda_init.sh
conda activate ffvt_u2n_p36
```