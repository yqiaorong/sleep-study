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
parser.add_argument('--layer_start_idx', default=None, type=int)
parser.add_argument('--num_feat',        default=3000, type=int)
parser.add_argument('--whiten',          default=False,type=bool)
args = parser.parse_args()

print('')
print(f'>>> Feature selection of sleemory images feature maps (resnext) for chunk layers <<<')
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
# Feature selection
# =============================================================================

# Load training fmaps
train_fmaps_dir = f'dataset/sleemory_localiser/dnn_feature_maps/full_feature_maps/ResNet/'
train_fmaps_list = os.listdir(train_fmaps_dir)

# Load test fmaps
test_fmaps_dir = f'dataset/sleemory_retrieval/dnn_feature_maps/full_feature_maps/ResNet/'
test_fmaps_list = os.listdir(test_fmaps_dir)

# Load layer names
sample_fmaps = scipy.io.loadmat(train_fmaps_dir+train_fmaps_list[0])
layers = list(sample_fmaps.keys())
del sample_fmaps

# Set the layer chunk size
layer_start_idx = args.layer_start_idx
layer_end_idx = min(layer_start_idx+50, len(layers))
print(layer_start_idx, layer_end_idx)


best_train_fmaps_all = {}
best_test_fmaps_all = {}
for ilayer, layer in enumerate(layers[3:][layer_start_idx:layer_end_idx]):
    print(layer)
    
    # Load test fmaps of one image
    for ifname, fname in enumerate(tqdm(test_fmaps_list, desc='test fmaps')):
        test_fmaps = scipy.io.loadmat(test_fmaps_dir+fname)
        # Select fmaps of the current layer
        if ifname == 0:
            test_fmaps_layer = test_fmaps[layer]
        else:
            test_fmaps_layer = np.concatenate((test_fmaps_layer, test_fmaps[layer]), axis=0)
        del test_fmaps
    print(f'Test fmaps shape: {test_fmaps_layer.shape}')
    
    # Load training fmaps of one image
    for ifname, fname in enumerate(tqdm(train_fmaps_list, desc='train fmaps')):
        train_fmaps = scipy.io.loadmat(train_fmaps_dir+fname)
        # Select fmaps of the current layer
        if ifname == 0:
            train_fmaps_layer = train_fmaps[layer]
        else:
            train_fmaps_layer = np.concatenate((train_fmaps_layer, train_fmaps[layer]), axis=0)
        del train_fmaps
    print(f'Train fmaps shape: {train_fmaps_layer.shape}')
    
    # Check if feature numbers <= num best features
    if test_fmaps_layer.shape[1] <= args.num_feat:
        pass
    else:
        # Build the feature selection model
        feature_selection = SelectKBest(f_regression, 
                                        k=args.num_feat).fit(train_fmaps_layer, train_eeg)
        # Select the best features
        train_fmaps_layer = feature_selection.transform(train_fmaps_layer)
        test_fmaps_layer = feature_selection.transform(test_fmaps_layer)
        print(f'The new train fmaps of {layer} has shape {train_fmaps_layer.shape}')
        print(f'The new test fmaps of {layer} has shape {test_fmaps_layer.shape}')
    
    best_train_fmaps_all[layer] = train_fmaps_layer
    best_test_fmaps_all[layer] = test_fmaps_layer
    del test_fmaps_layer, train_fmaps_layer
    print('')
    
# =============================================================================
# Save new features
# =============================================================================

# Save dir
train_save_dir = f'dataset/sleemory_localiser/dnn_feature_maps/best_feature_maps/ResNet'
if os.path.isdir(train_save_dir) == False:
    os.makedirs(train_save_dir)
    
test_save_dir = f'dataset/sleemory_retrieval/dnn_feature_maps/best_feature_maps/ResNet'
if os.path.isdir(test_save_dir) == False:
    os.makedirs(test_save_dir)

# Save
best_fmaps_fname = f'ResNet-best-{args.num_feat}-{layer_start_idx}_{whiten}fmaps.mat'
print(best_fmaps_fname)
scipy.io.savemat(f'{train_save_dir}/{best_fmaps_fname}', best_train_fmaps_all) 
scipy.io.savemat(f'{test_save_dir}/{best_fmaps_fname}', best_test_fmaps_all) 