# Dropout Reduces Underfitting

Official PyTorch implementation for **Dropout Reduces Underfitting**

> [**Dropout Reduces Underfitting**](https://arxiv.org/abs/2303.01500)<br>
> [Zhuang Liu*](https://liuzhuang13.github.io), [Zhiqiu Xu*](https://oscarxzq.github.io), [Joseph Jin](https://www.linkedin.com/in/joseph-jin/), [Zhiqiang Shen](https://zhiqiangshen.com/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/) (* equal contribution)
> <br>Meta AI, UC Berkeley and MBZUAI<br>

<p align="center">
<img src="https://user-images.githubusercontent.com/8370623/222586143-3500fa5b-c294-48c9-a5cf-5fac2659e519.png" width=50% height=50% 
class="center">
</p>

Figure: We propose **early dropout** and **late dropout**. Early dropout helps underfitting models fit the data better and achieve lower training loss. Late dropout helps improve the generalization performance of overfitting models.


## Results on ImageNet-1K

### Early Dropout

results with basic recipe (s.d. = stochastic depth)

| model| ViT-T | Mixer-S | Swin-F | ConvNeXt-F |
|:---|:---:|:---:|:---:|:---:|
| no dropout     | 73.9 | 71.0       | 74.3   | 76.1       |
| standard dropout   | 67.9  | 67.1       | 71.6   | -          |
| standard s.d. | 72.6  | 70.5       | 73.7   | 75.5       |
| early dropout     | **74.3**  | [**71.3**](https://drive.google.com/file/d/199i9rRD-u2DA22qmoZH774ibyhFUH_mE/view?usp=share_link)     | **74.7**   | -          |
| early s.d.    | **74.4**  | [**71.7**](https://drive.google.com/file/d/1jPtWufetAQhM4oe6wOgdTsKozYCXRmdb/view?usp=share_link)     | **75.2**   | **76.3**       |


results with improved recipe

| model        | ViT-T | Swin-F | ConvNeXt-F |
|:------------|:-----:|:------:|:----------:|
| no dropout     | 76.3  | 76.1   | 77.5       |
| standard dropout   | 71.5  | 73.5   | -          |
| standard s.d. | 75.6  | 75.6   | 77.4       |
| early dropout     | [**76.7**](https://drive.google.com/file/d/1q3kopfA2KazTaR9kuEEM5lzdNKHX2OQl/view?usp=share_link) | [**76.6**](https://drive.google.com/file/d/1Os16aIWD1WpSlccsFboesc0BgXN6KJ9C/view?usp=share_link) | -          |
| early s.d.    | [**76.7**](https://drive.google.com/file/d/1GTfGbNObvGDytdb9F5wgUHxnhlVRXs6o/view?usp=share_link) | [**76.6**](https://drive.google.com/file/d/17mNr8e-TVQoVM0I6IxVJNC4f3Y--T4R_/view?usp=share_link) | [**77.7**](https://drive.google.com/file/d/1sIePqyxk5ajVdsSCRCJoTIZsV8O_fKmO/view?usp=share_link)    |


### Late Dropout
results with basic recipe

| model        | ViT-B | Mixer-B |
|:------------:|:-----:|:-------:|
| standard s.d.   | 81.6  | 78.0    |
| late s.d.    | [**82.3**](https://drive.google.com/file/d/1_AB51g6AHF-C9oGWffwOw4C1Xug1LT_0/view?usp=share_link) | [**78.6**](https://drive.google.com/file/d/1CWEi8hyEIKz7F21HlsaEIgp8eaHNFHfe/view?usp=share_link)   |


## Installation
Please check [INSTALL.md](INSTALL.md) for installation instructions. 

## Training

### Basic Recipe
We list commands for early dropout, early stochastic depth on `ViT-T` and late stochastic depth on `ViT-B`.
- For training other models, change `--model` accordingly, e.g., to `vit_tiny`, `mixer_s32`, `convnext_femto`, `mixer_b16`, `vit_base`.
- Our results were produced with 4 nodes, each with 8 gpus. Below we give example commands on both multi-node and single-machine setups.

**Early dropout**

multi-node
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model vit_tiny --epochs 300 \
--batch_size 128 --lr 4e-3 --update_freq 1 \
--dropout 0.1 --drop_mode early --drop_schedule linear --cutoff_epoch 50 \
--data_path /path/to/data/ \
--output_dir /path/to/results/
```

single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model vit_tiny --epochs 300 \
--batch_size 128 --lr 4e-3 --update_freq 4 \
--dropout 0.1 --drop_mode early --drop_schedule linear --cutoff_epoch 50 \
--data_path /path/to/data/ \
--output_dir /path/to/results/
```

**Early stochastic depth**
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model vit_tiny --epochs 300 \
--batch_size 128 --lr 4e-3 --update_freq 4 \
--drop_path 0.5 --drop_mode early --drop_schedule linear --cutoff_epoch 50 \
--data_path /path/to/data/ \
--output_dir /path/to/results/
```

**Late stochastic depth**
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model vit_base --epochs 300 \
--batch_size 128 --lr 4e-3 --update_freq 4 \
--drop_path 0.4 --drop_mode late --drop_schedule constant --cutoff_epoch 50 \
--data_path /path/to/data/ \
--output_dir /path/to/results/
```

**Standard dropout / no dropout** (replace $p with 0.1 / 0.0)
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model vit_tiny --epochs 300 \
--batch_size 128 --lr 4e-3 --update_freq 4 \
--dropout $p --drop_mode standard \
--data_path /path/to/data/ \
--output_dir /path/to/results/
```


### Improved Recipe
Our improved recipe extends training epochs from `300` to `600`, and reduces both `mixup` and `cutmix` to `0.3`.

**Early dropout**
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model vit_tiny --epochs 600 --mixup 0.3 --cutmix 0.3 \
--batch_size 128 --lr 4e-3 --update_freq 4 \
--dropout 0.1 --drop_mode early --drop_schedule linear --cutoff_epoch 50 \
--data_path /path/to/data/ \
--output_dir /path/to/results/
```

**Early stochastic depth**
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model vit_tiny --epochs 600 --mixup 0.3 --cutmix 0.3 \
--batch_size 128 --lr 4e-3 --update_freq 4 \
--drop_path 0.5 --drop_mode early --drop_schedule linear --cutoff_epoch 50 \
--data_path /path/to/data/ \
--output_dir /path/to/results/
```

### Evaluation

single-GPU
```
python main.py --model vit_tiny --eval true \
--resume /path/to/model \
--data_path /path/to/data
```

multi-GPU
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model vit_tiny --eval true \
--resume /path/to/model \
--data_path /path/to/data
```

We will release ImageNet-1K model weights soon.

## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library and [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) codebase.

## License
This project is released under the CC-BY-NC 4.0 license. Please see the [LICENSE](LICENSE) file for more information.

## Citation
If you find this repository helpful, please consider citing:
```bibtex
@article{liu2023dropout,
  title={Dropout Reduces Underfitting},
  author={Zhuang Liu, Zhiqiu Xu, Joseph Jin, Zhiqiang Shen, Trevor Darrell},
  year={2023},
  journal={arXiv preprint arXiv:2303.01500},
}
```


