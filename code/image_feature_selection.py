import os
import argparse
import numpy as np
from tqdm import tqdm

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--layer_name', default='conv5', type=str)
parser.add_argument('--num_feat', default=300, type=int)
args = parser.parse_args()

print('')
print('Feature selection of THINGS2 images feature maps AlexNet <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220

# =============================================================================
# Load the feature maps
# =============================================================================

### Load the feature maps ###
feats = []
feats_all = []
# The dictionaries storing the dnn training feature maps in 3 stages
fmaps_train = {}
fmaps_train_1 = {}
fmaps_train_2 = {}
# The directory of the dnn training feature maps
fmaps_dir = os.path.join('dataset','THINGS_EEG2', 'dnn_feature_maps',
                        'full_feature_maps', 'Alexnet', 
                        'pretrained-'+str(args.pretrained),
                        'training_images')
fmaps_list = os.listdir(fmaps_dir)
fmaps_list.sort()
for f, fmaps in enumerate(tqdm(fmaps_list, desc='training_images')):
    fmaps_data = np.load(os.path.join(fmaps_dir, fmaps),
                            allow_pickle=True).item()
    all_layers = fmaps_data.keys()
    for l, dnn_layer in enumerate(all_layers):
        if f == 0:
            feats.append([[np.reshape(fmaps_data[dnn_layer], -1)]])
        else:
            feats[l].append([np.reshape(fmaps_data[dnn_layer], -1)])
    
fmaps_train[args.layer_name] = np.squeeze(np.asarray(feats[l]))
print('The original training fmaps shape', fmaps_train[args.layer_name].shape)

# =============================================================================
# Feature selection
# =============================================================================

from sklearn.feature_selection import SelectKBest, f_regression

### Load the training THINGS EEG2 data ###
# Load the THINGS2 training EEG data directory
eeg_train_dir = os.path.join('dataset', 'THINGS_EEG2', 'preprocessed_data')
# Iterate over THINGS2 subjects
eeg_data_train = []
for train_subj in tqdm(range(1,7), desc='THINGS EEG2 subjects'):
    # Load the THINGS2 training EEG data
    data = np.load(os.path.join(eeg_train_dir,'sub-'+format(train_subj,'02'),
                  'preprocessed_eeg_training.npy'), allow_pickle=True).item()
    # Get the THINGS2 training channels and times
    if train_subj == 1:
        train_ch_names = data['ch_names']
    else:
        pass
    # Average the training EEG data across repetitions: (16540,64,100)
    data = np.mean(data['preprocessed_eeg_data'], 1)
    # Average the training EEG data over time: (16540,64)
    data = np.mean(data, -1)
    # Average the training EEG data across electrodes: (16540,)
    data = np.mean(data, -1)
    # Append individual data
    eeg_data_train.append(data)
    del data
# Average the training EEG data across subjects: (16540,)
eeg_data_train = np.mean(eeg_data_train, 0)

# Fit feature selection model and transform
best_feat = SelectKBest(f_regression, k=args.num_feat).fit_transform(fmaps_train[args.layer_name], 
                                                        eeg_data_train)
del fmaps_train[args.layer_name]

# non_zero_count = np.count_nonzero(best_feat, axis=1)
# print(non_zero_count[:100])

# =============================================================================
# Save new features
# =============================================================================

# Create the saving directory if not existing
save_dir = os.path.join('dataset','THINGS_EEG2','dnn_feature_maps')
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)

# Save new features
np.save(os.path.join(save_dir, 'new_feature_maps'), best_feat) 