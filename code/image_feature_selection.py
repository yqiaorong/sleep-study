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
for train_subj in tqdm(range(1,11), desc='load THINGS EEG2 subjects'):
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
    # Drop the stimulus channel: (16540,63,100)
    data = np.delete(data, -1, axis=1)
    # Average the training EEG data over time: (16540,63)
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

# =============================================================================
# Save the feature selection model
# =============================================================================

save_dir = os.path.join('dataset','THINGS_EEG2')
pickle.dump(feature_selection, open(os.path.join(save_dir, 
                                    f'feat_model_{args.num_feat}.pkl'), 'wb'))

# =============================================================================
# Save new features
# =============================================================================

# Save new features
np.save(os.path.join(save_dir, 'dnn_feature_maps', 
                     f'new_feature_maps_{args.num_feat}'), best_feat) 