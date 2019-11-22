[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://github.com/Ha0Tang/AttentionGAN/blob/master/LICENSE.md)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
![Packagist](https://img.shields.io/badge/Pytorch-0.4.1-red.svg)
![Last Commit](https://img.shields.io/github/last-commit/Ha0Tang/AttentionGAN)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)]((https://github.com/Ha0Tang/AttentionGAN/graphs/commit-activity))
![Contributing](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)
![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)

# AttentionGAN for Unpaird Image-to-Image Translation

## AttentionGAN Framework
![Framework](./imgs/gesturegan_framework.jpg)

## Comparison with State-of-the-Art Image-to-Image Transaltion Methods
![Framework Comparison](./imgs/comparison.jpg)

### [Conference paper](https://arxiv.org/abs/1808.04859) | [Project page (Conference paper)](http://disi.unitn.it/~hao.tang/project/GestureGAN.html) | [Slides](http://disi.unitn.it/~hao.tang/uploads/slides/GestureGAN_MM18.pptx) | [Poster](http://disi.unitn.it/~hao.tang/uploads/posters/GestureGAN_MM18.pdf)

GestureGAN for Hand Gesture-to-Gesture Translation in the Wild.<br>
[Hao Tang](http://disi.unitn.it/~hao.tang/)<sup>1</sup>, [Wei Wang](https://weiwangtrento.github.io/)<sup>1,2</sup>, [Dan Xu](http://www.robots.ox.ac.uk/~danxu/)<sup>1,3</sup>, [Yan Yan](https://userweb.cs.txstate.edu/~y_y34/)<sup>4</sup> and [Nicu Sebe](http://disi.unitn.it/~sebe/)<sup>1</sup>. <br> 
<sup>1</sup>University of Trento, Italy, <sup>2</sup>EPFL, Switzerland, <sup>3</sup>University of Oxford, UK, <sup>4</sup>Texas State University, USA.<br>
In ACM MM 2018 (**Oral** & **Best Paper Candidate**).<br>
The repository offers the official implementation of our paper in PyTorch.

### [License](./LICENSE.md)

Copyright (C) 2019 University of Trento, Italy.

All rights reserved.
Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (**Attribution-NonCommercial-ShareAlike 4.0 International**)

The code is released for academic research use only. For commercial use, please contact [hao.tang@unitn.it](hao.tang@unitn.it).

## Installation

Clone this repo.
```bash
git clone https://github.com/Ha0Tang/GestureGAN
cd GestureGAN/
```

This code requires PyTorch 0.4.1 and python 3.6+. Please install dependencies by
```bash
pip install -r requirements.txt (for pip users)
```
or 

```bash
./scripts/conda_deps.sh (for Conda users)
```

To reproduce the results reported in the paper, you would need two NVIDIA GeForce GTX 1080 Ti GPUs or two NVIDIA TITAN Xp GPUs.

## Dataset Preparation

For hand gesture-to-gesture translation tasks, we use NTU Hand Digit and Creative Senz3D datasets.
For cross-view image translation task, we use Dayton and CVUSA datasets.
These datasets must be downloaded beforehand. Please download them on the respective webpages. In addition, we put a few sample images in this [code repo](https://github.com/Ha0Tang/GestureGAN/tree/master/datasets/samples). Please cite their papers if you use the data. 

**Preparing NTU Hand Digit Dataset**. The dataset can be downloaded in this [paper](https://rose.ntu.edu.sg/Publications/Documents/Action%20Recognition/Robust%20Part-Based%20Hand%20Gesture.pdf). After downloading it we adopt [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to generate hand skeletons and use them as training and testing data in our experiments. Note that we filter out failure cases in hand gesture estimation for training and testing. Please cite their papers if you use this dataset. Train/Test splits for Creative Senz3D dataset can be downloaded from [here](https://github.com/Ha0Tang/GestureGAN/tree/master/datasets/ntu_split). Download images and the crossponding extracted hand skeletons of this dataset:
```bash
bash ./datasets/download_gesturegan_dataset.sh ntu_image_skeleton
```
Then run the following MATLAB script to generate training and testing data:
```bash
cd datasets/
matlab -nodesktop -nosplash -r "prepare_ntu_data"
```

**Preparing Creative Senz3D Dataset**. The dataset can be downloaded [here](https://lttm.dei.unipd.it//downloads/gesture/#senz3d). After downloading it we adopt [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to generate hand skeletons and use them as training data in our experiments. Note that we filter out failure cases in hand gesture estimation for training and testing. Please cite their papers if you use this dataset. Train/Test splits for Creative Senz3D dataset can be downloaded from [here](https://github.com/Ha0Tang/GestureGAN/tree/master/datasets/senz3d_split). Download images and the crossponding extracted hand skeletons of this dataset:
```bash
bash ./datasets/download_gesturegan_dataset.sh senz3d_image_skeleton
```
Then run the following MATLAB script to generate training and testing data:
```bash
cd datasets/
matlab -nodesktop -nosplash -r "prepare_senz3d_data"
```

**Preparing Dayton Dataset**. The dataset can be downloaded [here](https://github.com/lugiavn/gt-crossview). In particular, you will need to download dayton.zip. 
Ground Truth semantic maps are not available for this datasets. We adopt [RefineNet](https://github.com/guosheng/refinenet) trained on CityScapes dataset for generating semantic maps and use them as training data in our experiments. Please cite their papers if you use this dataset.
Train/Test splits for Dayton dataset can be downloaded from [here](https://github.com/Ha0Tang/SelectionGAN/tree/master/datasets/dayton_split).

**Preparing CVUSA Dataset**. The dataset can be downloaded [here](https://drive.google.com/drive/folders/0BzvmHzyo_zCAX3I4VG1mWnhmcGc), which is from the [page](http://cs.uky.edu/~jacobs/datasets/cvusa/). After unzipping the dataset, prepare the training and testing data as discussed in [SelectionGAN](https://arxiv.org/abs/1904.06807). We also convert semantic maps to the color ones by using this [script](https://github.com/Ha0Tang/SelectionGAN/blob/master/scripts/convert_semantic_map_cvusa.m).
Since there is no semantic maps for the aerial images on this dataset, we use black images as aerial semantic maps for placehold purposes.

**Preparing Your Own Datasets**. Each training sample in the dataset will contain {Ix,Iy,Cx,Cy}, where Ix=image x, Iy=image y, Cx=Controllable structure of image x, and Cy=Controllable structure of image y.
Of course, you can use GestureGAN for your own datasets and tasks, such landmark-guided facial experssion translation and keypoint-guided person image generation.

## Generating Images Using Pretrained Model

Once the dataset is ready. The result images can be generated using pretrained models.

1. You can download a pretrained model (e.g. ntu_gesturegan_twocycle) with the following script:

```
bash ./scripts/download_gesturegan_model.sh ntu_gesturegan_twocycle
```
The pretrained model is saved at `./checkpoints/[type]_pretrained`. Check [here](https://github.com/Ha0Tang/GestureGAN/blob/master/scripts/download_gesturegan_model.sh) for all the available GestureGAN models.

2. Generate images using the pretrained model.
```bash
python test.py --dataroot [path_to_dataset] \
  --name [type]_pretrained \
  --model [gesturegan_model] \
  --which_model_netG resnet_9blocks \
  --which_direction AtoB \
  --dataset_mode aligned \
  --norm batch \
  --gpu_ids 0 \
  --batchSize [BS] \
  --loadSize [LS] \
  --fineSize [FS] \
  --no_flip
```

`[path_to_dataset]` is the path to the dataset. Dataset can be one of `ntu`, `senz3d`, `dayton_a2g`, `dayton_g2a` and `cvusa`. `[type]_pretrained` is the directory name of the checkpoint file downloaded in Step 1, which should be one of `ntu_gesturegan_twocycle_pretrained`, `senz3d_gesturegan_twocycle_pretrained`, `dayton_a2g_64_gesturegan_onecycle_pretrained`, `dayton_g2a_64_gesturegan_onecycle_pretrained`, `dayton_a2g_gesturegan_onecycle_pretrained`, `dayton_g2a_gesturegan_onecycle_pretrained` and `cvusa_gesturegan_onecycle_pretrained`. 
`[gesturegan_model]` is the directory name of the model of GestureGAN, which should be one of `gesturegan_twocycle` or `gesturegan_onecycle`.
If you are running on CPU mode, change `--gpu_ids 0` to `--gpu_ids -1`. For [`BS`, `LS`, `FS`], please see `Training` and `Testing` sections.

Note that testing requires large amount of disk storage space. If you don't have enough space, append `--saveDisk` on the command line.
    
3. The outputs images are stored at `./results/[type]_pretrained/` by default. You can view them using the autogenerated HTML file in the directory.

## Training New Models

New models can be trained with the following commands.

1. Prepare dataset. 

2. Train.

For NTU dataset:
```bash
export CUDA_VISIBLE_DEVICES=3,4;
python train.py --dataroot ./datasets/ntu \
  --name ntu_gesturegan_twocycle \
  --model gesturegan_twocycle \
  --which_model_netG resnet_9blocks \
  --which_direction AtoB \
  --dataset_mode aligned \
  --norm instance \
  --gpu_ids 0,1 \
  --batchSize 4 \
  --loadSize 286 \
  --fineSize 256 \
  --no_flip \
  --lambda_L1 800 \
  --cyc_L1 0.1 \
  --lambda_identity 0.01 \
  --lambda_feat 1000 \
  --display_id 0 \
  --niter 10 \
  --niter_decay 10
```

For Senz3D dataset:
```bash
export CUDA_VISIBLE_DEVICES=5,7;
python train.py --dataroot ./datasets/senz3d \
  --name senz3d_gesturegan_twocycle \
  --model gesturegan_twocycle \
  --which_model_netG resnet_9blocks \
  --which_direction AtoB \
  --dataset_mode aligned \
  --norm instance \
  --gpu_ids 0,1 \
  --batchSize 4 \
  --loadSize 286 \
  --fineSize 256 \
  --no_flip \
  --lambda_L1 800 \
  --cyc_L1 0.1 \
  --lambda_identity 0.01 \
  --lambda_feat 1000 \
  --display_id 0 \
  --niter 10 \
  --niter_decay 10
```

For CVUSA dataset:
```bash
export CUDA_VISIBLE_DEVICES=0;
python train.py --dataroot ./dataset/cvusa \
  --name cvusa_gesturegan_onecycle \
  --model gesturegan_onecycle \
  --which_model_netG resnet_9blocks \
  --which_direction AtoB \
  --dataset_mode aligned \
  --norm instance \
  --gpu_ids 0 \
  --batchSize 4 \
  --loadSize 286 \
  --fineSize 256 \
  --no_flip \
  --cyc_L1 0.1 \
  --lambda_identity 100 \
  --lambda_feat 100 \
  --display_id 0 \
  --niter 15 \
  --niter_decay 15
```

For Dayton (a2g direction, 256) dataset:
```bash
export CUDA_VISIBLE_DEVICES=0;
python train.py --dataroot ./datasets/dayton_a2g \
  --name dayton_a2g_gesturegan_onecycle \
  --model gesturegan_onecycle \
  --which_model_netG resnet_9blocks \
  --which_direction AtoB \
  --dataset_mode aligned \
  --norm instance \
  --gpu_ids 0 \
  --batchSize 4 \
  --loadSize 286 \
  --fineSize 256 \
  --no_flip \
  --cyc_L1 0.1 \
  --lambda_identity 100 \
  --lambda_feat 100 \
  --display_id 0 \
  --niter 20 \
  --niter_decay 15
```

For Dayton (g2a direction, 256) dataset:
```bash
export CUDA_VISIBLE_DEVICES=1;
python train.py --dataroot ./datasets/dayton_g2a \
  --name dayton_g2a_gesturegan_onecycle \
  --model gesturegan_onecycle \
  --which_model_netG resnet_9blocks \
  --which_direction AtoB \
  --dataset_mode aligned \
  --norm instance \
  --gpu_ids 0 \
  --batchSize 4 \
  --loadSize 286 \
  --fineSize 256 \
  --no_flip \
  --cyc_L1 0.1 \
  --lambda_identity 100 \
  --lambda_feat 100 \
  --display_id 0 \
  --niter 20 \
  --niter_decay 15
```

For Dayton (a2g direction, 64) dataset:
```bash
export CUDA_VISIBLE_DEVICES=0;
python train.py --dataroot ./datasets/dayton_a2g \
  --name dayton_a2g_64_gesturegan_onecycle \
  --model gesturegan_onecycle \
  --which_model_netG resnet_9blocks \
  --which_direction AtoB \
  --dataset_mode aligned \
  --norm instance \
  --gpu_ids 0 \
  --batchSize 16 \
  --loadSize 72 \
  --fineSize 64 \
  --no_flip \
  --cyc_L1 0.1 \
  --lambda_identity 100 \
  --lambda_feat 100 \
  --display_id 0 \
  --niter 50 \
  --niter_decay 50
```

For Dayton (g2a direction, 64) dataset:
```bash
export CUDA_VISIBLE_DEVICES=1;
python train.py --dataroot ./datasets/dayton_g2a \
  --name dayton_g2a_64_gesturegan_onecycle \
  --model gesturegan_onecycle \
  --which_model_netG resnet_9blocks \
  --which_direction AtoB \
  --dataset_mode aligned \
  --norm instance \
  --gpu_ids 0 \
  --batchSize 16 \
  --loadSize 72 \
  --fineSize 64 \
  --no_flip \
  --cyc_L1 0.1 \
  --lambda_identity 100 \
  --lambda_feat 100 \
  --display_id 0 \
  --niter 50 \
  --niter_decay 50
```

There are many options you can specify. Please use `python train.py --help`. The specified options are printed to the console. To specify the number of GPUs to utilize, use `export CUDA_VISIBLE_DEVICES=[GPU_ID]`. Note that train `gesturegan_onecycle` only needs one GPU, while train `gesturegan_twocycle` needs two GPUs.

To view training results and loss plots on local computers, set `--display_id` to a non-zero value and run `python -m visdom.server` on a new terminal and click the URL [http://localhost:8097](http://localhost:8097/).
On a remote server, replace `localhost` with your server's name, such as [http://server.trento.cs.edu:8097](http://server.trento.cs.edu:8097).

### Can I continue/resume my training? 
To fine-tune a pre-trained model, or resume the previous training, use the `--continue_train --which_epoch <int> --epoch_count<int+1>` flag. The program will then load the model based on epoch `<int>` you set in `--which_epoch <int>`. Set `--epoch_count <int+1>` to specify a different starting epoch count.


## Testing

Testing is similar to testing pretrained models.

For NTU dataset:
```bash
python test.py --dataroot ./datasets/ntu \
  --name ntu_gesturegan_twocycle \
  --model gesturegan_twocycle \
  --which_model_netG resnet_9blocks \
  --which_direction AtoB \
  --dataset_mode aligned \
  --norm instance \
  --gpu_ids 0 \
  --batchSize 4 \
  --loadSize 286 \
  --fineSize 256 \
  --no_flip
```

For Senz3D dataset:
```bash
python test.py --dataroot ./datasets/senz3d \
  --name senz3d_gesturegan_twocycle \
  --model gesturegan_twocycle \
  --which_model_netG resnet_9blocks \
  --which_direction AtoB \
  --dataset_mode aligned \
  --norm instance \
  --gpu_ids 0 \
  --batchSize 4 \
  --loadSize 286 \
  --fineSize 256 \
  --no_flip
```

For CVUSA dataset:
```bash
python test.py --dataroot ./datasets/cvusa \
  --name cvusa_gesturegan_onecycle \
  --model gesturegan_onecycle \
  --which_model_netG resnet_9blocks \
  --which_direction AtoB \
  --dataset_mode aligned \
  --norm instance \
  --gpu_ids 0 \
  --batchSize 4 \
  --loadSize 286 \
  --fineSize 256 \
  --no_flip
```

For Dayton (a2g direction, 256) dataset:
```bash
python test.py --dataroot ./datasets/dayton_a2g \
  --name dayton_a2g_gesturegan_onecycle \
  --model gesturegan_onecycle \
  --which_model_netG resnet_9blocks \
  --which_direction AtoB \
  --dataset_mode aligned \
  --norm instance \
  --gpu_ids 0 \
  --batchSize 4 \
  --loadSize 286 \
  --fineSize 256 \
  --no_flip
```

For Dayton (g2a direction, 256) dataset:
```bash
python test.py --dataroot ./datasets/dayton_g2a \
  --name dayton_g2a_gesturegan_onecycle \
  --model gesturegan_onecycle \ 
  --which_model_netG resnet_9blocks \
  --which_direction AtoB \
  --dataset_mode aligned \
  --norm instance \
  --gpu_ids 0 \
  --batchSize 4 \
  --loadSize 286 \
  --fineSize 256 \
  --no_flip
```

For Dayton (a2g direction, 64) dataset:
```bash
python test.py --dataroot ./datasets/dayton_a2g \
  --name dayton_g2a_64_gesturegan_onecycle \
  --model gesturegan_onecycle \
  --which_model_netG resnet_9blocks \
  --which_direction AtoB \
  --dataset_mode aligned \
  --norm instance \
  --gpu_ids 0 \
  --batchSize 16 \
  --loadSize 72 \
  --fineSize 64 \
  --no_flip
```

For Dayton (g2a direction, 64) dataset:
```bash
python test.py --dataroot ./datasets/dayton_g2a \
  --name dayton_g2a_64_gesturegan_onecycle \
  --model gesturegan_onecycle \
  --which_model_netG resnet_9blocks \
  --which_direction AtoB \
  --dataset_mode aligned \
  --norm instance \
  --gpu_ids 0 \
  --batchSize 16 \
  --loadSize 72 \
  --fineSize 64 \
  --no_flip
```

Use `--how_many` to specify the maximum number of images to generate. By default, it loads the latest checkpoint. It can be changed using `--which_epoch`.

## Code Structure

- `train.py`, `test.py`: the entry point for training and testing.
- `models/gesturegan_onecycle_model.py`, `models/gesturegan_twocycle_model.py`: creates the networks, and compute the losses.
- `models/networks/`: defines the architecture of all models for GestureGAN.
- `options/`: creates option lists using `argparse` package.
- `data/`: defines the class for loading images and controllable structures.
- `scripts/evaluation`: several evaluation codes.

## Evaluation Code

We use several metrics to evaluate the quality of the generated images:

## Ecaluation Code
- [FID](https://github.com/bioinf-jku/TTUR): Official Implementation
- [KID](https://github.com/taki0112/GAN_Metrics-Tensorflow): Suggested by [UGATIT](https://github.com/taki0112/UGATIT/issues/64). 
  Install Steps: `conda create -n python36 pyhton=3.6 anaconda` and `pip install --ignore-installed --upgrade tensorflow==1.13.1`

## Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{tang2018gesturegan,
  title={GestureGAN for Hand Gesture-to-Gesture Translation in the Wild},
  author={Tang, Hao and Wang, Wei and Xu, Dan and Yan, Yan and Sebe, Nicu},
  booktitle={ACM MM},
  year={2018}
}
```

## Acknowledgments
This source code is inspired by [Pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). We want to thank the NVIDIA Corporation for the donation of the TITAN Xp GPUs used in this work.

## Related Projects (Controllable Image-to-Image Translation)

### Keypoint/Skeleton Guided Person/Gesture Image Generation
Person
- [Dense Intrinsic Appearance Flow for Human Pose Transfer (CVPR 2019, PyTorch)](https://github.com/ly015/intrinsic_flow)
- [Progressive Pose Attention for Person Image Generation (CVPR 2019, PyTorch)](https://github.com/tengteng95/Pose-Transfer)
- [Unsupervised Person Image Generation with Semantic Parsing Transformation (CVPR 2019, PyTorch)](https://github.com/SijieSong/person_generation_spt)
- [Pose-Normalized Image Generation for Person Re-identification (ECCV 2018, PyTorch)](https://github.com/naiq/PN_GAN)
- [Everybody Dance Now (ECCVW 2018, PyTorch)](https://github.com/nyoki-mtl/pytorch-EverybodyDanceNow)
- [FD-GAN: Pose-guided Feature Distilling GAN for Robust Person Re-identification (NIPS 2018, PyTorch)](https://github.com/yxgeee/FD-GAN)
- [Disentangled Person Image Generation (CVPR 2018, Tensorflow)](https://github.com/charliememory/Disentangled-Person-Image-Generation)
- [Deformable GANs for Pose-Based Human Image Generation (CVPR 2018, Tensorflow)](https://github.com/AliaksandrSiarohin/pose-gan)
- [Pose Guided Person Image Generation (NIPS 2017, Tensorflow)](https://github.com/charliememory/Pose-Guided-Person-Image-Generation)

Gesture
- [Gesture-to-Gesture Translation in the Wild via Category-Independent Conditional Maps (ACM MM 2019, PyTorch)](https://github.com/yhlleo/TriangleGAN)

### Label/Landmark Guided Facial Image Generation
- [Few-Shot Adversarial Learning of Realistic Neural Talking Head Models (PyTorch)](https://github.com/grey-eye/talking-heads)
- [GANimation: Anatomically-aware Facial Animation from a Single Image (ECCV 2018, PyTorch)](https://github.com/albertpumarola/GANimation)
- [StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation (CVPR 2018, PyTorch)](https://github.com/yunjey/stargan)

### Semantic Map Guided Cross-View Image Translation
- [Multi-Channel Attention Selection GAN with Cascaded Semantic Guidance for Cross-View Image Translation (CVPR 2019, PyTorch)](https://github.com/Ha0Tang/SelectionGAN)
- [Cross-View Image Synthesis using Conditional GANs (CVPR 2018, Torch)](https://github.com/kregmi/cross-view-image-synthesis)
- [Predicting Ground-Level Scene Layout from Aerial Imagery (CVPR 2017, Tensorflow)](https://github.com/viibridges/crossnet)

## Contributions
If you have any questions/comments/bug reports, feel free to open a github issue or pull a request or e-mail to the author Hao Tang ([hao.tang@unitn.it](hao.tang@unitn.it)).

