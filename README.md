# Multimodal Transformer with Multi-View Visual Representation for Image Captioning

This repository corresponds to the PyTorch implementation of [Multimodal Transformer with Multi-View Visual Representation for Image Captioning](https://arxiv.org/abs/1905.07841v1). By using the commonly used bottom-up-attention visual features, a single svbase model delivers 130.9 Cider on the Kapathy's test split of MSCOCO dataset. Please check our paper for details.

## Table of Contents

0. [Prerequisites](#Prerequisites)
1. [Training](#Training)
2. [Testing](#Testing)

## Prerequisites

#### Requirements

- [Python 3](https://www.python.org/downloads/)
- [PyTorch](http://pytorch.org/) >= 1.1
- [Cuda](https://developer.nvidia.com/cuda-toolkit) >= 9.0 and [cuDNN](https://developer.nvidia.com/cudnn)

The annotations files can be downloaded [here](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/ES91VBvL885MvEVSXeozrXEBRdeQcvj0OplbE2ujooMylQ?e=mRSClL) and unzipped to the datasets folder.

The bottom up features can be extracted by ours [bottom-up-attention](https://github.com/MILVLG/bottom-up-attention.pytorch) repo.

Finally, the datasets folders will have the following structure:

```angular2html
|-- datasets
   |-- mscoco
   |  |-- features
   |  |  |-- frcn-r101
   |  |  |  |-- train2014
   |  |  |  |  |-- COCO_train2014_....npz
   |  |  |  |-- val2014
   |  |  |  |  |-- COCO_val2014_....npz
   |  |  |  |-- test2015
   |  |  |  |  |-- COCO_test2015_....npz
   |  |-- annotations
   |  |  |-- coco-train-idxs.p
   |  |  |-- coco-train-words.p
   |  |  |-- cocotalk_label.h5
   |  |  |-- cocotalk.json
   |  |  |-- vocab.json
   |  |  |-- glove_embeding.npy
```

## Training

The following script will train a model with cross-entropy loss :

```bash
$ python train.py --caption_model svbase --ckpt_path <checkpoint_dir> --gpu_id 0
```

1. `caption_model` refers to the model while been trained, such as svbase, umv, umv3.

2. `ckpt_path` refers to the dir to save checkpoint.

3. `gpu_id` refers to the gpu id.

Based on the model trained with cross-entropy loss, the following script will load the pre-trained model and then fine-tune the model with self-critical loss:

```bash
$ python train.py --caption_model svbase --learning_rate 1e-5 --ckpt_path <checkpoint_dir> --start_from <checkpoint_dir_rl> --gpu_id 0 --max_epochs 25
```

1. `caption_model` refers to the model while been trained.

2. `learning_rate` refers to the learning rate use in self-critical.

3. `ckpt_path` refers to the dir to save checkpoint.

4. `gpu_id` refers to the gpu id.

## Testing

Given the trained model, the following script will test the performance on the `val` split of MSCOCO:

```bash
$ python eval.py --model <checkpoint_dir>/model-best.pth --infos_path <checkpoint_dir>/infos.pkl --gpu_id 0
```

1. `model` refers to the path of model's checkpoint file.

2. `infos_path` refers to the path of model's informations file.

3. `gpu_id` refers to the gpu id.

## Citation

```
@article{yu2019multimodal,
  title={Multimodal transformer with multi-view visual representation for image captioning},
  author={Yu, Jun and Li, Jing and Yu, Zhou and Huang, Qingming},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2019},
  publisher={IEEE}
}
```

## Acknowledgement
We thank Ruotian Luo for his [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch), [cider](https://github.com/ruotianluo/cider/tree/e9b736d038d39395fa2259e39342bb876f1cc877) and [coco-caption](https://github.com/ruotianluo/coco-caption/tree/ea20010419a955fed9882f9dcc53f2dc1ac65092) repos.