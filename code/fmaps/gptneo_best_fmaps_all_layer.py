import os
import torch
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy
from sklearn.feature_selection import SelectKBest, f_regression

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--num_feat',default=1000, type=int)
parser.add_argument('--whiten',  default=False,type=bool)
args = parser.parse_args()

print('')
print(f'>>> Sleemory images best feature maps (gptneo) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')

# =============================================================================
# Load eeg data
# =============================================================================

# Set up
if args.whiten == False:
    whiten = ''
else:
    whiten = 'whiten_'
     

eeg_fname = f'unique_{whiten}eeg.mat'
print(eeg_fname)

# Load localiser EEG
train_eeg = scipy.io.loadmat(f'output/sleemory_localiser/whiten_eeg/{eeg_fname}')
train_eeg = train_eeg[f'{whiten}eeg']
# Average across time to shape (img, ch,)
train_eeg = np.mean(train_eeg, -1)
# Average across channel to shape (img,)
train_eeg = np.mean(train_eeg, -1)
print('Training (localiser) EEG data shape (img,)', train_eeg.shape)

# =============================================================================
# Load fmaps
# =============================================================================

# Load training fmaps
train_fmaps = scipy.io.loadmat('dataset/sleemory_localiser/dnn_feature_maps/full_feature_maps/GPTNeo/gptneo_fmaps.mat')
for ilayer, layer in enumerate(list(train_fmaps.keys())[3:]):
    if ilayer == 0:
        train_fmaps_all = train_fmaps[layer]
    else:
        train_fmaps_all = np.concatenate((train_fmaps_all, train_fmaps[layer]), axis=1)
print(f'train fmaps all shape: {train_fmaps_all.shape}') 
    
# Load training fmaps
test_fmaps = scipy.io.loadmat('dataset/sleemory_retrieval/dnn_feature_maps/full_feature_maps/GPTNeo/gptneo_fmaps.mat')
for ilayer, layer in enumerate(list(test_fmaps.keys())[3:]):
    print(layer)
    if ilayer == 0:
        test_fmaps_all = test_fmaps[layer]
    else:
        test_fmaps_all = np.concatenate((test_fmaps_all, test_fmaps[layer]), axis=1)
print(f'test fmaps all shape: {test_fmaps_all.shape}')  

# =============================================================================
# Feature selection
# =============================================================================
    
# Build the feature selection model
feature_selection = SelectKBest(f_regression, 
                                k=args.num_feat).fit(train_fmaps_all, train_eeg)
# Select the best features
train_fmaps_all = feature_selection.transform(train_fmaps_all)
test_fmaps_all = feature_selection.transform(test_fmaps_all)
print(f'The final train fmaps has shape {train_fmaps_all.shape}')
print(f'The final test fmaps has shape {test_fmaps_all.shape}')

# =============================================================================
# Save new features
# =============================================================================

# Save dir
train_save_dir = f'dataset/sleemory_localiser/dnn_feature_maps'
if os.path.isdir(train_save_dir) == False:
    os.makedirs(train_save_dir)
    
test_save_dir = f'dataset/sleemory_retrieval/dnn_feature_maps'
if os.path.isdir(test_save_dir) == False:
    os.makedirs(test_save_dir)

# Save
best_fmaps_fname = f'GPTNEO-best-{args.num_feat}_{whiten}fmaps.mat'
print(best_fmaps_fname)
scipy.io.savemat(f'{train_save_dir}/{best_fmaps_fname}', {'fmaps': train_fmaps_all}) 
scipy.io.savemat(f'{test_save_dir}/{best_fmaps_fname}', {'fmaps': test_fmaps_all})