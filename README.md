# mt-captioning

This repository corresponds to the PyTorch implementation of the paper *Multimodal Transformer with Multi-View Visual Representation for Image Captioning*. By using the bottom-up-attention visual features (with slight improvement), our single-view Multimodal Transformer model (MT_sv) delivers 130.9 CIDEr on the Kapathy's test split of MSCOCO dataset. Please check our [paper](https://arxiv.org/abs/1905.07841v1) for details.

## Table of Contents

0. [Prerequisites](#Prerequisites)
1. [Training](#Training)
2. [Testing](#Testing)

## Prerequisites

#### Requirements

- [Python 3](https://www.python.org/downloads/)
- [PyTorch](http://pytorch.org/) >= 1.4.0
- [Cuda](https://developer.nvidia.com/cuda-toolkit) >= 9.2 and [cuDNN](https://developer.nvidia.com/cudnn)

The annotation files can be downloaded [here](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/ES91VBvL885MvEVSXeozrXEBRdeQcvj0OplbE2ujooMylQ?e=mRSClL) and unzipped to the datasets folder.

The visual features are extracted by our [bottom-up-attention.pytorch](https://github.com/MILVLG/bottom-up-attention.pytorch) repo using the following scripts:

```bash
# 1.extract the bbox from the image
$ python3 extract_features.py --mode caffe \
          --config-file configs/bua-caffe/extract-bua-caffe-r101-bbox-only.yaml \
          --image-dir <image_dir> --out-dir <bbox_dir> --resume

# 2. extract the roi feature by bbox
$ python3 extract_features.py --mode caffe \
         --config-file configs/bua-caffe/extract-bua-caffe-r101-gt-bbox.yaml \
         --image-dir <image_dir> --gt-bbox-dir <bbox_dir> --out-dir <output_dir> --resume
```
We provided a pre-extracted features in the `datasets/mscoco/features/val2014` folder for the image in `datasets/mscoco/image` to help validating the correctness of the extracted features.

We use the ResNet-101 as our backbone and extract features for the MSCOCO dataset to the `datasets/mscoco/features/frcn-r101` folder.

Finally, the `datasets` folder will have the following structure:

```angular2html
|-- datasets
   |-- mscoco
   |  |-- features
   |  |  |-- frcn-r101
   |  |  |  |-- train2014
   |  |  |  |  |-- COCO_train2014_....jpg.npz
   |  |  |  |-- val2014
   |  |  |  |  |-- COCO_val2014_....jpg.npz
   |  |  |  |-- test2015
   |  |  |  |  |-- COCO_test2015_....jpg.npz
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

1. `caption_model` refers to the model while been trained, such as `svbase` and `umv`

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

Given the trained model, the following script will report the performance on the `val` split of MSCOCO:

```bash
$ python test.py --ckpt_path <checkpoint_dir> --gpu_id 0
```

1. `ckpt_path` refers to the dir to save checkpoint.

2. `gpu_id` refers to the gpu id.

## Pre-trained models

We provided the pre-trained model for the single-view MT model at present. More models will be added in the future.

Model |  Backbone  | BLEU@1 | METEOR | CIDEr |Download
:-:|:-:|:-:|:-:|:-:|:-:
MT_sv|ResNet-101|80.8|29.1|130.9|[model](https://awma1-my.sharepoint.com/:u:/g/personal/yuz_l0_tn/EUakyWWLZ7dGkoO_bljASwABpKPNgKARbuiAyQvaA6dDYg?e=mQYtBy)

## Citation

If this repository is helpful for your research, we'd really appreciate it if you could cite the following paper:
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
