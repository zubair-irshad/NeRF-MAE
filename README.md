<!-- # NeRF-MAE: Masked AutoEncoders for Self-Supervised 3D Representation Learning for Neural Radiance Fields -->

<div align="center">
    <img src="demo/nerf-mae_teaser.png" alt="Voltron Logo"/>
    <img src="demo/nerf-mae_teaser.jpeg" width="100%">
</div>
<!-- <p align="center">
<img src="demo/nerf-mae_teaser.jpeg" width="100%">
</p> -->

<br>
<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2404.01300-gray?style=for-the-badge&logo=arxiv&logoColor=white&color=B31B1B)](https://arxiv.org/abs/2404.01300)
[![Project Page](https://img.shields.io/badge/Project-Page-orange?style=for-the-badge&logoColor=white&labelColor=gray&link=https%3A%2F%2Fnerf-mae.github.io%2F)](https://nerf-mae.github.io)
[![Pytorch](https://img.shields.io/badge/Pytorch-%3E1.12-gray?style=for-the-badge&logo=pytorch&logoColor=white&labelColor=gray&color=ee4c2c&link=https%3A%2F%2Fnerf-mae.github.io%2F)](https://pytorch.org/)
[![Cite](https://img.shields.io/badge/Cite-Bibtex-gray?style=for-the-badge&logoColor=white&color=F7A41D
)](https://github.com/zubair-irshad/NeRF-MAE?tab=readme-ov-file#citation)
</div>

---

<a href="https://www.tri.global/" target="_blank">
 <img align="right" src="demo/GeorgiaTech_RGB.png" width="18%"/>
</a>

<a href="https://www.tri.global/" target="_blank">
 <img align="right" src="demo/tri-logo.png" width="17%"/>
</a>

This repository is the pytorch implementation of our **ECCV 2024** paper, **NeRF-MAE**


**NeRF-MAE: Masked AutoEncoders for Self-Supervised 3D Representation Learning for Neural Radiance Fields**

<a href="https://zubairirshad.com"><strong>Muhammad Zubair Irshad</strong></a>
·
<a href="https://zakharos.github.io/"><strong>Sergey Zakharov</strong></a>
·
<a href="https://www.linkedin.com/in/vitorguizilini"><strong>Vitor Guizilini</strong></a>
·
<a href="https://adriengaidon.com/"><strong>Adrien Gaidon</strong></a>
·
<a href="https://faculty.cc.gatech.edu/~zk15/"><strong>Zsolt Kira</strong></a>
·
<a href="https://www.tri.global/about-us/dr-rares-ambrus"><strong>Rares Ambrus</strong></a>
<br> **European Conference on Computer Vision, ECCV 2024**<br>

<b> Toyota Research Institute &nbsp; | &nbsp; Georgia Institute of Technology</b>

## 💡 Highlights
- **NeRF-MAE**: The first large-scale pretraining utilizing Neural Radiance Fields (NeRF) as an input modality. We pretrain a single Transformer model on thousands of NeRFs for 3D representation learning.
- **NeRF-MAE Dataset**: A large-scale NeRF pretraining and downstream task finetuning dataset.

## 🏷️ TODO 

- [x] Release training code
- [x] Release NeRF-MAE dataset comprising radiance and density grids
- [ ] Pretrained NeRF-MAE checkpoints and out-of-the-box model usage
- [ ] Release multi-view rendered images and Instant-NGP checkpoints (totalling 1.6M+ posed images and 3200+ trained NeRF checkpoints)

## NeRF-MAE Model Architecture
<p align="center">
<img src="demo/nerf-mae_architecture.jpg" width="90%">
</p>

<!-- _________________ 


<p align="center">
<img src="demo/comparison_mae.jpeg" width="100%">
</p> -->

## Citation

If you find this repository or our dataset useful, please consider citing:

```
@inproceedings{irshad2024nerfmae,
    title={NeRF-MAE: Masked AutoEncoders for Self-Supervised 3D Representation Learning for Neural Radiance Fields},
    author={Muhammad Zubair Irshad and Sergey Zakharov and Vitor Guizilini and Adrien Gaidon and Zsolt Kira and Rares Ambrus},
    journal={European Conference on Computer Vision (ECCV)},
    year={2024}
    }
```

### Contents
 - [🌇  Environment](#-environment)
 - [⛳ Dataset](#-dataset)
 - [💫 Usage (Coming Soon)](#-usage)
 - [📉 Pretraining](#-pretraining)
 - [📊 Finetuning](#-finetuning)
 - [📌 FAQ](#-faq)

 ## 🌇  Environment

Create a python 3.7 virtual environment and install requirements:

```bash
cd $NeRF-MAE repo
conda create -n nerf-mae python=3.9
conda activate nerf-mae
pip install --upgrade pip
pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```
The code was built and tested on **cuda 11.3**

## ⛳ Dataset

Download the preprocessed datasets here. 

- Pretraining dataset (comprising NeRF radiance and density grids). [Download link]()
- Finetuning dataset (comprising NeRF radiance and density grids and bounding box/semantic labelling annotations). [3D Object Detection](), [3D Semantic Segmentation (Coming Soon)](), [Voxel-Super Resolution (Coming Soon)]()


Note: The above datasets are all you need to train and evaluate our method. Bonus: we will be releasing our multi-view rendered posed RGB images from FRONT3D, HM3D and Hypersim as well as Instant-NGP trained checkpoints soon (these comprise over 1.6M+ images and 3200+ NeRF checkpoints)

Please note that our dataset was generated using the instruction from [NeRF-RPN]([NeRF-RPN](https://github.com/lyclyc52/NeRF_RPN)). Please consider citing both our work and NeRF-RPN if you find this dataset useful in your research. 

## 💫 Usage (Coming Soon)

NeRF-MAE (package: nerf-mae) is structured to provide easy access to pretrained NeRF-MAE models (and reproductions), to facilitate use for various downstream tasks. Our pretraining provides an easy-to-access embedding of any NeRF scene, which can be used for a variety of downstream tasks in a straightforwaed way.

Using a pretrained NeRF-MAE model is easy:

Navigate to **nerf-mae** folder and run pretraining script. 

## 📉 Pretraining

Ofcourse, you can also pretrain your own NeRF-MAE models. Navigate to **nerf-mae** folder and run pretraining script. 

```
cd nerf-mae
bash train_mae3d.sh
```

Checkout **train_mae3d.sh** file for a complete list of all hyperparameters such as ```num_epochs```, ```lr```, ```masking_prob``` etc. 


## 📊 Finetuning

Our finetuning code is largely based on [NeRF-RPN](https://github.com/lyclyc52/NeRF_RPN). Infact, we use the same dataset as NeRF-RPN (unseen during pretraining), for finetuning. This makes sure our comparison with NeRF-RPN is based on the same architecture, the only difference is the network weights are started from scratch for NeRF-RPN, whereas in our case, we start with our pretrained network weights. Please see our paper for more details.

**Note**: We do not see ScanNet dataset during our pretraining. ScanNet 3D OBB prediction finetuning is a challenging case of cross-dataset transfer. 


### 3D Object Detection
Navigate to **nerf-rpn** folder and run finetuning script. 

To run finetuning with our pretrained weights:

```
cd nerf-rpn
bash train_fcos_pretrained.sh
```

To run finetuning with starting from scratch weights:

```
cd nerf-rpn
bash train_fcos.sh
```

### 3D Semantic Segmentation
Navigate to **nerf-rpn** folder and run finetuning script. 

To run finetuning with our pretrained weights:

```
cd nerf-rpn
bash train_fcos_pretrained.sh
```

To run finetuning with starting from scratch weights:

```
cd nerf-rpn
bash train_fcos.sh
```

### 3D Voxel Super-Resolution
Navigate to **nerf-rpn** folder and run finetuning script. 

To run finetuning with our pretrained weights:

```
cd nerf-rpn
bash train_fcos_pretrained.sh
```

To run finetuning with starting from scratch weights:

```
cd nerf-rpn
bash train_fcos.sh
```

Checkout **train_fcos_pretraining.sh** file for a complete list of all hyperparameters such as ```mae_checkpoint```, ```num_epochs```, ```lr```, ```masking_prob``` etc. 

## Acknowledgments
This code is built upon the implementation from [NeRF-RPN](https://github.com/lyclyc52/NeRF_RPN).

## Licenses
This repository and dataset is released under the [CC BY-NC 4.0](https://github.com/zubair-irshad/NeO-360/blob/master/LICENSE.md) license.
