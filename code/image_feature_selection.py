import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from func import load_full_fmaps

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
# Load the training feature maps
# =============================================================================

fmaps_train = load_full_fmaps(args)

# =============================================================================
# Feature selection
# =============================================================================

from sklearn.feature_selection import SelectKBest, f_regression

### Load the training THINGS EEG2 data ###
# Load the THINGS2 training EEG data directory
eeg_train_dir = os.path.join('dataset', 'THINGS_EEG2', 'preprocessed_data')
# Iterate over THINGS2 subjects
eeg_data_train = []
for train_subj in tqdm(range(1,11), desc='THINGS EEG2 subjects'):
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
feature_selection = SelectKBest(f_regression, k=args.num_feat).fit(fmaps_train[args.layer_name], 
                                                        eeg_data_train)
best_feat = feature_selection.transform(fmaps_train[args.layer_name])

print(f'The new training fmaps shape {best_feat.shape}')
del fmaps_train[args.layer_name]

# non_zero_count = np.count_nonzero(best_feat, axis=1)
# print(non_zero_count[:100])

# =============================================================================
# Save the feature selection model
# =============================================================================

save_dir = os.path.join('dataset','THINGS_EEG2')
pickle.dump(feature_selection, open(os.path.join(save_dir, 'feat_model.pkl'), 'wb'))

# =============================================================================
# Save new features
# =============================================================================

# Create the saving directory if not existing
save_dir = os.path.join('dataset','THINGS_EEG2','dnn_feature_maps')
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)

# Save new features
np.save(os.path.join(save_dir, 'new_feature_maps'), best_feat) 