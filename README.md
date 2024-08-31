# Data Augmentation through Self-Distillation for Fine-Grained Visual Recognition in Low Data Regimes


## Approach

<p align="center"> 
<img src="docs/Images/architecture.png" width="750">
</p>

> **<p align="justify"> Abstract:** *The emerging task of fine-grained image classification in low-data regimes assumes the presence of low inter-class variance and large intra-class variation along with a highly limited amount of training samples per class available. In this paper, we present a novel approach, designed to enhance deep neural network performance on this challenge. Specifically, we propose a framework, called AD-Net, leveraging the power of Augmentation and Distillation techniques. Our approach is able to refine learned features through self-distillation on augmented samples, mitigating harmful overfitting. In detail, our AD-Net incorporates a multi-branch configuration with shared weights, enabling efficient knowledge transfer and feature consolidation. We provide an extensive study on the impact of distillation architecture, augmentation types, and objective functions in our framework, highlighting the importance of each in achieving optimal performance. We conduct comprehensive experiments on popular fine-grained image classification benchmarks and demonstrate consistent improvement over state-of-the-art low-data techniques based on ResNet-50 model, providing an absolute improvement of up to 10% using the lowest data regimes. AD-Net establishes a significant step forward in low-data fine-grained image classification, showcasing promising results and insights for future research in this domain. Well-documented source code and trained models will be publicly released* </p>


## Main Contributions
1) We introduce a simple yet effective  architecture-independent approach to improve the performance of models at FGVC under low data regime.
2) Our proposed low-data framework includes two feature distillation branches for model purification both using randomly cropped inputs, which provides explicit regularisation while not relying on any extra module.
3) Our extensive experiments on three popular FGVC datasets (CUB, Stanford Cars, and FGVC-Aircraft) demonstrate that our approach achieves state-of-the-art results on FGVC in low data regimes.
4) Our solution allows the model to learn and refine representations by enriching the variability of the training data through extra augmented image views.


<hr />


# 🐘 Model Zoo

### Main Models
(Soon)

#### Experimental Models (outside the paper)
(Soon)


<hr />


# 🧋 How to start

## Installation
(Soon)

For environment installation and pre-trained models preparation, please follow the instructions in [INSTALL.md](docs/INSTALL.md). 


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

## Data preparation
(Soon)

For datasets preparation, please follow the instructions in [DATASET.md](docs/DATASET.md).

## Training and Evaluation
(Soon)

For training and evaluation, please follow the instructions in [RUN.md](docs/RUN.md).

#### Run:

1. Initialise the code space:

```
cd '/path/to/cloned/repo/'

bash init.sh 

conda activate ffvt_u2n_p36
```

2. Run (example):

Ours:
```
python3 -W ignore train.py --dataset CUB --model_type cnn --model_name resnet50 --img_size 224 --resize_size 256 --train_batch_size 24 --eval_batch_size 24 --learning_rate 0.03 --num_steps 40000 --fp16 --eval_every 200 --split 10 --aug_type double_crop
```

Vanilla:
```
python3 -W ignore train.py --dataset CUB --model_type cnn --model_name resnet50 --img_size 224 --resize_size 256 --train_batch_size 24 --eval_batch_size 24 --learning_rate 0.03 --num_steps 40000 --fp16 --eval_every 200 --split 10 --lr_ratio 1.0 --vanilla
```


## Dev:

```
bash init.sh 

module load cuda-10.2 # no need anymore
nvcc --version # no need anymore
```

### Use local environment:
```
cd '/l/users/20020067/Activities/FGIC/low_data/Combined/Refine/fgic_lowd'

source /apps/local/anaconda2023/conda_init.sh
conda activate ffvt_u2n_p36_timm # old: ffvt_u2n_p36 (messed up torch and timm)
```

### Use shared environment:
```
cd '/path/to/cloned/repo/'

source /apps/local/anaconda2023/conda_init.sh
conda activate /l/users/cv-805/envs/ffvt_u2n_p36
```

### Misc:

#### Copy environment to the shared folder:

```
conda create --prefix /<path>/<target_name> --clone <source_name>
```

```
CUB_ft_05LossFocal01_resnet50vanilla_1k_224_bs24_doubleCrop2564n6481_sepNfix_autoSched05_LrRatio10_rn50_40k_noSAMtrashVanilla_doubleAugs224_KLlossSAM001_batchmean_inputLog_lr003_ld10_test
```


<hr />


# 🆕 News
* **(Nov 30, 2023)** 
  * Repo description added (README.md).
  * Training and evaluation code is released.

* **(Soon)** 
  * Pretrained models will be released.
  * Code instructions will be added (INSTALL.md, DATASET.md, RUN.md).
  * Optimisation

<hr />


# 🖋️ Credits

## Citation
In case you would like to utilise or refer to our approach (source code, trained models, or results) in your research, please consider citing:

```
(Soon)
```

## Contacts
In case you have a question or suggestion, please create an issue or contact us at _dmitry.demidov@mbzuai.ac.ae_ .

## Acknowledgements
Our code is partially based on [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch) repository and we thank the corresponding authors for releasing their code. If you use our derived code, please consider giving credits to these works as well.
