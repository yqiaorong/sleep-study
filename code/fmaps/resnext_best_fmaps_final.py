import os
import argparse
import numpy as np
import scipy
from tqdm import tqdm
from sklearn.feature_selection import SelectKBest, f_regression

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--new_num_feat', default=3000, type=int)
parser.add_argument('--old_num_feat', default=1000, type=int)
parser.add_argument('--whiten',       default=False,type=bool)
args = parser.parse_args()

print('')
print(f'>>> Feature selection of sleemory images feature maps (resnext) final round <<<')
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


# file list
layer_start_indices = range(0, 287, 50)
file_list = [f'ResNet-best-{args.old_num_feat}-{idx}_{whiten}fmaps.mat' for idx in layer_start_indices]

# =============================================================================
# Load training feature maps
# =============================================================================

train_fmaps_dir = f'dataset/sleemory_localiser/dnn_feature_maps/best_feature_maps/ResNet'
for ichunk, chunk in enumerate(tqdm(file_list, desc='train chunk file')):
    train_fmaps = scipy.io.loadmat(train_fmaps_dir+chunk)
    
    # Concatenate layers in chunk file
    for ilayer, layer in enumerate(list(train_fmaps.keys())[3:]):
        if ilayer == 0:
            train_fmaps_chunk = train_fmaps[layer]
        else:
            train_fmaps_chunk = np.concatenate((train_fmaps_chunk, train_fmaps[layer]), axis=1)
    print(f'train fmaps chunk shape: {train_fmaps_chunk.shape}')
    del train_fmaps
    
    # Concatenate chunk files
    if ichunk == 0:
        train_fmaps_all = train_fmaps_chunk
    else:
        train_fmaps_all = np.concatenate((train_fmaps_all, train_fmaps_chunk), axis=1)
    del train_fmaps_chunk
print(f'train fmaps all shape: {train_fmaps_all.shape}')  

# =============================================================================
# Load test feature maps
# =============================================================================

test_fmaps_dir = f'dataset/sleemory_retrieval/dnn_feature_maps/best_feature_maps/ResNet'
for ichunk, chunk in enumerate(tqdm(file_list, desc='train chunk file')):
    test_fmaps = scipy.io.loadmat(test_fmaps_dir+chunk)
    
    # Concatenate layers in chunk file
    for ilayer, layer in enumerate(list(test_fmaps.keys())[3:]):
        if ilayer == 0:
            test_fmaps_chunk = test_fmaps[layer]
        else:
            test_fmaps_chunk = np.concatenate((test_fmaps_chunk, test_fmaps[layer]), axis=1)
    print(f'test fmaps chunk shape: {test_fmaps_chunk.shape}')
    del test_fmaps
    
    # Concatenate chunk files
    if ichunk == 0:
        test_fmaps_all = test_fmaps_chunk
    else:
        test_fmaps_all = np.concatenate((test_fmaps_all, test_fmaps_chunk), axis=1)
    del test_fmaps_chunk
print(f'test fmaps all shape: {test_fmaps_all.shape}')  
    
# =============================================================================
# Feature selection
# =============================================================================

# Build the feature selection model
feature_selection = SelectKBest(f_regression, 
                                k=args.new_num_feat).fit(train_fmaps_all, train_eeg)
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
best_fmaps_fname = f'ResNet-best-{args.new_num_feat}_{whiten}fmaps.mat'
print(best_fmaps_fname)
scipy.io.savemat(f'{train_save_dir}/{best_fmaps_fname}', train_fmaps_all) 
scipy.io.savemat(f'{test_save_dir}/{best_fmaps_fname}', test_fmaps_all)