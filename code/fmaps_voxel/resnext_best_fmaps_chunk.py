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
eeg_fname = f'sub-{args.sub:03d}_task-localiser_source_data'

eeg_data = mat73.loadmat(os.path.join(eeg_dir, eeg_fname+'.mat'))
eeg = eeg_data['sub_eeg_loc']['eeg']

eeg_labels = eeg_data['sub_eeg_loc']['images']
eeg_labels = [s[0].split('.')[0] for s in eeg_labels]
eeg_labels = np.asarray(eeg_labels)

    
# Average across time to shape (img, ch,)
eeg = np.mean(eeg, -1)
# Average across channel to shape (img,)
eeg = np.mean(eeg, -1)
print('Training (localiser) EEG data shape (img,)', eeg.shape)
del eeg_data

# =============================================================================
# Feature selection
# =============================================================================

# Load localiser fmaps
fmaps_dir = f'dataset/sleemory_localiser/dnn_feature_maps/full_feature_maps/ResNet/'
fmaps_labels = eeg_labels

# Load retrieval fmaps
retri_fmaps_dir = f'dataset/sleemory_retrieval/dnn_feature_maps/full_feature_maps/ResNet/'
retri_fmaps_labels = os.listdir(retri_fmaps_dir)
retri_fmaps_labels = [s.split('.')[0] for s in retri_fmaps_labels]
retri_fmaps_labels = np.asarray(retri_fmaps_labels)



# Load layer names
sample_fmaps = scipy.io.loadmat(fmaps_dir+fmaps_labels[0]+'_feats.mat')
layers = list(sample_fmaps.keys())

# Set the layer chunk size
layer_start_idx = args.layer_start_idx
layer_end_idx = min(layer_start_idx+args.layer_idx_num, len(layers))
print(layer_start_idx, layer_end_idx)



best_fmaps_all = {}
retri_best_fmaps_all = {}
for ilayer, layer in enumerate(layers[3:][layer_start_idx:layer_end_idx]):
    print(layer)
    
    # Load training fmaps of one image
    fmaps_layer = np.empty((fmaps_labels.shape[0], sample_fmaps[layer].shape[1]))
    print(fmaps_layer.shape)
    for ifname, fname in enumerate(tqdm(fmaps_labels, desc='concatenate train fmaps per layer')):
        fmaps = scipy.io.loadmat(fmaps_dir+fname+'_feats.mat')
        fmaps_layer[ifname] = fmaps[layer]
        del fmaps
    print(f'Selected localiser fmaps shape: {fmaps_layer.shape}')
    
    # Check if feature numbers <= num best features
    if fmaps_layer.shape[1] <= args.num_feat:
        pass
    else:
        # Build the feature selection model
        feature_selection = SelectKBest(f_regression, 
                                        k=args.num_feat).fit(fmaps_layer, eeg)
        # Select the best features
        fmaps_layer = feature_selection.transform(fmaps_layer)
        print(f'The new fmaps of {layer} has shape {fmaps_layer.shape}')
    
    best_fmaps_all[layer] = fmaps_layer
    del fmaps_layer
    print('')
    
    # Load retrieval fmaps of one image
    retri_fmaps_layer = np.empty((retri_fmaps_labels.shape[0], sample_fmaps[layer].shape[1]))
    print(retri_fmaps_layer.shape)
    for ifname, fname in enumerate(tqdm(retri_fmaps_labels, desc='concatenate test fmaps per layer')):
        retri_fmaps = scipy.io.loadmat(retri_fmaps_dir+fname+'.mat')
        retri_fmaps_layer[ifname] = retri_fmaps[layer]
        del retri_fmaps
    print(f'Retrieval fmaps shape: {retri_fmaps_layer.shape}')
    
    # Check if feature numbers <= num best features
    if retri_fmaps_layer.shape[1] <= args.num_feat:
        pass
    else:
        # Select the best features
        retri_fmaps_layer = feature_selection.transform(retri_fmaps_layer)
        print(f'The new fmaps of {layer} has shape {retri_fmaps_layer.shape}')
    
    retri_best_fmaps_all[layer] = retri_fmaps_layer
    del retri_fmaps_layer
    print('')

best_fmaps_all['imgs_all'] = fmaps_labels
retri_best_fmaps_all['imgs_all'] = retri_fmaps_labels
    
# =============================================================================
# Save new features
# =============================================================================

# Save dir
train_save_dir = f'output/sleemory_localiser_vox/dnn_feature_maps/best_feature_maps/sub_{args.sub}/ResNet/'
if os.path.isdir(train_save_dir) == False:
    os.makedirs(train_save_dir)

test_save_dir = f'output/sleemory_retrieval_vox/dnn_feature_maps/best_feature_maps/sub_{args.sub}/ResNet/'
if os.path.isdir(test_save_dir) == False:
    os.makedirs(test_save_dir)

# Save
best_fmaps_fname = f'ResNet-best-{args.num_feat}-{layer_start_idx}_fmaps.mat'
print(best_fmaps_fname)

scipy.io.savemat(f'{train_save_dir}/{best_fmaps_fname}', best_fmaps_all) 
print('localiser fmaps saved.')

scipy.io.savemat(f'{test_save_dir}/{best_fmaps_fname}', retri_best_fmaps_all) 
print('retrieval fmaps saved.')
