from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
import eval_utils
import argparse
import misc.utils as utils
import torch
from misc.beamSearchEvalEnsemble import *

# Input arguments and options
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default='0',
                    help='gpu id')
parser.add_argument('--cnn_model', type=str,  default='resnet101',
                help='resnet101, resnet152')
parser.add_argument('--batch_size', type=int, default=1,
                help='if > 0 then overrule, otherwise load from checkpoint.')
parser.add_argument('--num_images', type=int, default=-1,
                help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--language_eval', type=int, default=0,
                help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--dump_images', type=int, default=0,
                help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
parser.add_argument('--dump_json', type=int, default=1,
                help='Dump json with predictions into vis folder? (1=yes,0=no)')
parser.add_argument('--dump_path', type=int, default=0,
                help='Write image paths along with predictions into vis json? (1=yes,0=no)')

# Sampling options
parser.add_argument('--sample_max', type=int, default=1,
                help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--beam_size', type=int, default=5,
                help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
parser.add_argument('--temperature', type=float, default=1.0,
                help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
# For evaluation on a folder of images:
parser.add_argument('--image_folder', type=str, default='',
                help='If this is nonempty then will predict on the images in this folder path')
parser.add_argument('--image_root', type=str, default='',
                help='In case the image paths have to be preprended with a root path to an image folder')
# For evaluation on MSCOCO images from some split:
parser.add_argument('--input_fc_dir', type=str, default='/data/features/mscoco/detfeat_resnet152',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_att_dir', type=str, default='/data/features/mscoco/detfeat_resnet101_bbox101',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_label_h5', type=str, default='data/cocotalk-glove_label.h5',
                help='path to the h5file containing the preprocessed dataset')
parser.add_argument('--input_json', type=str, default='data/cocotalk-glove.json',
                help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
parser.add_argument('--split', type=str, default='test',
                help='if running on MSCOCO images, which split to use: val|test|train')
parser.add_argument('--coco_json', type=str, default='data/instances_val2014.json',
                help='if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')
# misc
parser.add_argument('--id', type=str, default='',
                help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
# ensemble
parser.add_argument('--models', nargs='+',type=str, default='log_transformer_rl2/model-best.pth',
                help='path to model to evaluate')
parser.add_argument('--infos_paths', nargs='+',type=str, default='log_transformer_rl2/infos_transformer-best.pkl',
                help='path to infos to evaluate')

opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

model_paths = opt.models
infos_paths = opt.infos_paths


with open(infos_paths[0]) as f:
    infos = cPickle.load(f)

vocab = infos['vocab'] # ix -> word mapping

# Setup the model

model_list = []

for i in range(len(model_paths)):
    # Load infos
    # override and collect parameters
    with open(infos_paths[i]) as f:
        infos_model = cPickle.load(f)

    opt = parser.parse_args()
    if len(opt.input_att_dir) == 0:
        opt.input_fc_dir = infos_model['opt'].input_fc_dir
        opt.input_att_dir = infos_model['opt'].input_att_dir
        opt.input_label_h5 = infos_model['opt'].input_label_h5
    if len(opt.input_json) == 0:
        opt.input_json = infos_model['opt'].input_json
    if opt.batch_size == 0:
        opt.batch_size = infos_model['opt'].batch_size
    if len(opt.id) == 0:
        opt.id = infos_model['opt'].id
    ignore = ["id", "batch_size", "beam_size", "start_from", "language_eval", "input_att_dir", "gpu_id", "input_fc_dir", "input_json"]
    for k in vars(infos_model['opt']).keys():
        if k not in ignore:
            if k in vars(opt):
                assert vars(opt)[k] == vars(infos_model['opt'])[k], k + ' option not consistent'
            else:
                vars(opt).update({k: vars(infos_model['opt'])[k]}) # copy over options from model

    model = models.setup(opt)
    model.load_state_dict(torch.load(model_paths[i]))
    model.cuda()
    model.eval()
    model_list.append(model)

# Create the Data Loader instance
loader = DataLoader(opt)
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos['vocab']


# Set sample options
split_predictions, lang_stats = eval_utils.eval_split_ensemble(model_list, loader,
    vars(opt))

if lang_stats:
  print(lang_stats)

if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open('vis/'+opt.split+'.json', 'w'))


