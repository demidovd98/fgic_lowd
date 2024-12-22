# Extract More from Less: Efficient Fine-Grained Visual Recognition in Low-Data Regimes

Official repository for the paper "[Extract More from Less: Efficient Fine-Grained Visual Recognition in Low-Data Regimes](https://bmvc2024.org/proceedings/859/)", <br>
accepted as a Full Paper to [BMVC '24](https://bmvc2024.org/).

> [**Extract More from Less: Efficient Fine-Grained Visual Recognition in Low-Data Regimes**](https://bmvc2024.org/proceedings/859/)
> [![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://bmva-archive.org.uk/bmvc/2024/papers/Paper_859/paper.pdf)<br>
> [Dmitry Demidov](https://scholar.google.es/citations?hl=en&pli=1&user=k3euI0sAAAAJ)


## Approach

<p align="center"> 
<img src="docs/Images/architecture.png" width="750">
</p>

> **<p align="justify"> Abstract:** *The emerging task of fine-grained image classification in low-data regimes assumes the presence of low inter-class variance and large intra-class variation along with a highly limited amount of training samples per class. However, traditional ways of separately dealing with fine-grained categorisation and extremely scarce data may be inefficient under both these harsh conditions presented together. In this paper, we present a novel framework, called AD-Net, aiming to enhance deep neural network performance on this challenge by leveraging the power of Augmentation and Distillation techniques. Specifically, our approach is designed to refine learned features through self-distillation on augmented samples, mitigating harmful overfitting. We conduct comprehensive experiments on popular fine-grained image classification benchmarks where our AD-Net demonstrates consistent improvement over traditional fine-tuning and state-of-the-art low-data techniques. Remarkably, with the smallest data available, our framework shows an outstanding relative accuracy increase of up to 45 % compared to standard ResNet-50 and up to 27 % compared to the closest SOTA runner-up. We emphasise that our approach is practically architecture-independent and adds zero extra cost at inference time. Additionally, we provide an extensive study on the impact of every framework’s component, highlighting the importance of each in achieving optimal performance.* </p>


## Main Contributions
1) We introduce a simple yet effective  architecture-independent approach to improve the performance of models at FGVC under the low-data regime.
2) Our proposed low-data framework includes two feature distillation branches for model purification both using randomly cropped inputs, which provides explicit regularisation while not relying on any extra module.
3) Our extensive experiments on three popular FGVC datasets (CUB, Stanford Cars, and FGVC-Aircraft) demonstrate that our approach achieves state-of-the-art results on FGVC in low data regimes.
4) Our solution allows the model to learn and refine representations by enriching the variability of the training data through extra augmented image views.


<hr />


# 🐘 Model Zoo

### Main Models

The pre-trained models can be found here: [models](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/dmitry_demidov_mbzuai_ac_ae/EnMOkgG7VQlKmvBOndirZ3IBc-L9u0uwWBYjszZn8YHOOw?e=T9y1B1)



<hr />


# 🧋 How to start

## Installation

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


## Data Preparation

Put the datasets into the corresponding folders in the '/datasets/' directory.

Splits for low-data regimes (10-100 %) are provided for each dataset in the corresponding directory: '/datasets/{dataset}/low_data/'.

In case you put data in a different directory, you may need to change the path to datastes in the code.


## Training and Evaluation


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

<hr />


# 🆕 News
* **(Nov 16, 2024)** 
  * Repo description added (README.md).
  * Training and evaluation code is released.
  * Code instructions are added.

* **(Dec 7, 2024)** 
  * Pretrained models are released.

* **(Soon)** 
  * Optimisation

<hr />


# 🖋️ Credits

## Citation
In case you would like to utilise or refer to our approach (source code, trained models, or results) in your research, please consider citing:

```
@inproceedings{Demidov_2024_BMVC,
 author    = {Dmitry Demidov and Abduragim Shtanchaev and Mihail Minkov Mihaylov and Mohammad Almansoori},
 title     = {Extract More from Less: Efficient Fine-Grained Visual Recognition in Low-Data Regimes},
 booktitle = {35th British Machine Vision Conference 2024, {BMVC} 2024, Glasgow, UK, November 25-28, 2024},
 publisher = {BMVA},
 year      = {2024},
 url       = {https://papers.bmvc2024.org/0859.pdf}
}
```

## Contacts
In case you have a question or suggestion, please create an issue or contact us at _dmitry.demidov@mbzuai.ac.ae_ .

## Acknowledgements
Our code is partially based on [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch) repository and we thank the corresponding authors for releasing their code. If you use our derived code, please consider giving credits to these works as well.
