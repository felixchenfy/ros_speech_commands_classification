# -*- coding: future_fstrings -*-
#!/usr/bin/env python2
from __future__ import division
from __future__ import print_function

if 1: # Set path
    import sys, os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
    sys.path.append(ROOT)
    
import numpy as np 
import torch 

if 1: # my lib
    import utils.lib_io as lib_io
    import utils.lib_commons as lib_commons
    import utils.lib_datasets as lib_datasets
    import utils.lib_augment as lib_augment
    import utils.lib_ml as lib_ml
    import utils.lib_rnn as lib_rnn
    
# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------

# Set arguments ------------------------- 
args = lib_rnn.set_default_args()

args.num_epochs = 15
args.learning_rate = 0.001
args.train_eval_test_ratio=[0.7, 0.3, 0.0]
args.do_data_augment = True
args.data_folder = "data/data_train/"
args.classes_txt = "config/classes.names" 
args.load_weight_from = "weights/kaggle.ckpt"
args.finetune_model = True # If true, fix all parameters except the fc layer
args.save_model_to = 'checkpoints/' # Save model and log file

# Dataset -------------------------- 

# Get data's filenames and labels
file_paths, file_labels = lib_datasets.AudioDataset.load_classes_and_data_filenames(
    args.classes_txt, args.data_folder)

if 0: # DEBUG: use only a subset of all data
    GAP = 20
    file_paths = file_paths[::GAP]
    file_labels = file_labels[::GAP]
    args.num_epochs = 5
    
# Set data augmentation
if args.do_data_augment:
    Aug = lib_augment.Augmenter # rename
    aug = Aug([        
        Aug.Shift(rate=0.2, keep_size=False), 
        Aug.PadZeros(time=(0, 0.3)),
        Aug.Amplify(rate=(0.5, 1.2)),
        # Aug.PlaySpeed(rate=(0.7, 1.3), keep_size=False),
        # Aug.Noise(noise_folder="data/noises/", prob_noise=0.8, intensity=(0.1, 0.4)),
        #       There is already strong white noise in most of my data. No need to add noise.
    ], prob_to_aug=0.8)
else:
    aug = None

# Split data into train/eval/test
tr_X, tr_Y, ev_X, ev_Y, te_X, te_Y = lib_ml.split_train_eval_test(
    X=file_paths, Y=file_labels, ratios=args.train_eval_test_ratio, dtype='list')
train_dataset = lib_datasets.AudioDataset(file_paths=tr_X, file_labels=tr_Y, transform=aug)
eval_dataset = lib_datasets.AudioDataset(file_paths=ev_X, file_labels=ev_Y, transform=None)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=args.batch_size, shuffle=True)

# Create model and train -------------------------------------------------
model = lib_rnn.create_RNN_model(args, load_weight_from=args.load_weight_from) # create model
lib_rnn.train_model(model, args, train_loader, eval_loader)
