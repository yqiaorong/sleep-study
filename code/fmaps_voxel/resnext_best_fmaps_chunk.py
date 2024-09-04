import os
import argparse
import numpy as np
import scipy
from tqdm import tqdm
from sklearn.feature_selection import SelectKBest, f_regression
import mat73

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=2, type=int)
parser.add_argument('--layer_start_idx', default=None, type=int)
parser.add_argument('--layer_idx_num',   default=None, type=int)
parser.add_argument('--num_feat',        default=3000, type=int)
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

# Load localiser EEG
eeg_dir = '/home/simon/Documents/gitrepos/shannon_encodingmodelsEEG/dataset/sleemory_localiser/preprocessed_data'

fname = f'sub-{args.sub:03d}_task-localiser_source_data'
data = mat73.loadmat(os.path.join(eeg_dir, fname+'.mat'))
train_eeg = data['sub_eeg_loc']['eeg']

imgs_all = data['sub_eeg_loc']['images']
imgs_all = [s[0].split('.')[0] for s in imgs_all]
imgs_all = np.asarray(imgs_all)
    
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

# Load layer names
sample_fmaps = scipy.io.loadmat(train_fmaps_dir+train_fmaps_list[0])
layers = list(sample_fmaps.keys())
del sample_fmaps

# Set the layer chunk size
layer_start_idx = args.layer_start_idx
layer_end_idx = min(layer_start_idx+args.layer_idx_num, len(layers))
print(layer_start_idx, layer_end_idx)


best_train_fmaps_all = {}
best_test_fmaps_all = {}
for ilayer, layer in enumerate(layers[3:][layer_start_idx:layer_end_idx]):
    print(layer)
    
    # Load training fmaps of one image
    for ifname, fname in enumerate(tqdm(train_fmaps_list, desc='concatenate train fmaps per layer')):
        train_fmaps = scipy.io.loadmat(train_fmaps_dir+fname)
        # Select fmaps of the current layer
        if ifname == 0:
            train_fmaps_layer = train_fmaps[layer]
        else:
            train_fmaps_layer = np.concatenate((train_fmaps_layer, train_fmaps[layer]), axis=0)
        del train_fmaps
    print(f'Train fmaps shape: {train_fmaps_layer.shape}')
    
    # Check if feature numbers <= num best features
    if train_fmaps_layer.shape[1] <= args.num_feat:
        pass
    else:
        # Build the feature selection model
        feature_selection = SelectKBest(f_regression, 
                                        k=args.num_feat).fit(train_fmaps_layer, train_eeg)
        # Select the best features
        train_fmaps_layer = feature_selection.transform(train_fmaps_layer)
        print(f'The new train fmaps of {layer} has shape {train_fmaps_layer.shape}')
    
    best_train_fmaps_all[layer] = train_fmaps_layer
    del train_fmaps_layer
    print('')
    
# =============================================================================
# Save new features
# =============================================================================

# Save dir
train_save_dir = f'output/sleemory_localiser_vox/dnn_feature_maps/best_feature_maps/sub_{args.sub}/ResNet/'
if os.path.isdir(train_save_dir) == False:
    os.makedirs(train_save_dir)

# Save
best_fmaps_fname = f'ResNet-best-{args.num_feat}-{layer_start_idx}_fmaps.mat'
print(best_fmaps_fname)
scipy.io.savemat(f'{train_save_dir}/{best_fmaps_fname}', best_train_fmaps_all) 
print('Train fmaps saved.')