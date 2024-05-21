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
parser.add_argument('--pretrained', default=True,    type=bool)
# parser.add_argument('--layer_name', default='conv5', type=str)
parser.add_argument('--num_feat',   default=1000,    type=int)
parser.add_argument('--adapt_to',   default='',      type=str) # [/_sleemory]
args = parser.parse_args()

print('')
print('Feature selection of THINGS2 images feature maps <<<')
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
# Save directory
# =============================================================================

save_dir = os.path.join('dataset','THINGS_EEG2')

# =============================================================================
# Feature selection
# =============================================================================

if args.num_feat == -1:
    best_feat = fmaps_train
else:
    best_feat = {}
    
    from sklearn.feature_selection import SelectKBest, f_regression

    # =============================================================================
    # Load the training THINGS EEG2 data
    # =============================================================================
    
    # Load the THINGS2 training EEG data directory
    eeg_train_dir = os.path.join('dataset', 'THINGS_EEG2', 
                                'preprocessed_data'+args.adapt_to)
    # Iterate over THINGS2 subjects
    eeg_data_train = []
    for train_subj in tqdm(range(1,11), desc='load THINGS EEG2 subjects'):
        # Load the THINGS2 training EEG data
        if args.adapt_to == '':
            data = np.load(os.path.join(eeg_train_dir,'sub-'+format(train_subj,'02'),
                    'preprocessed_eeg_training.npy'), allow_pickle=True).item()
        else:
            data = np.load(os.path.join(eeg_train_dir,'sub-'+format(train_subj,'02'),
                    'preprocessed_eeg_training.npy'), allow_pickle=True)
        # Get the THINGS2 training channels and times
        if train_subj == 1:
            train_ch_names = data['ch_names']
        else:
            pass
        # Average the training EEG data across repetitions: (img, ch, time)
        data = np.mean(data['preprocessed_eeg_data'], 1)
        # Drop the stimulus channel: (img, ch, time)
        data = np.delete(data, -1, axis=1)
        # Average the training EEG data over time: (img, ch)
        data = np.mean(data, -1)
        # Average the training EEG data across electrodes: (img,)
        data = np.mean(data, -1)
        # Append individual data
        eeg_data_train.append(data)
        del data
    # Average the training EEG data across subjects: (img,)
    eeg_data_train = np.mean(eeg_data_train, 0)

   
    # layers names
    all_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
    # Apply feature selection
    for layer in all_layers:
        feature_selection = SelectKBest(f_regression, 
                                        k=args.num_feat).fit(fmaps_train[layer], 
                                        eeg_data_train)
        best_feat[layer] = feature_selection.transform(fmaps_train[layer])

        print(f'The new training fmaps of layer {layer} has shape {best_feat[layer].shape}')

        # =============================================================================
        # Save the feature selection model
        # =============================================================================
        
        model_dir = os.path.join(save_dir, 'model', 'best_fmaps_model')
        if os.path.isdir(model_dir) == False:
            os.makedirs(model_dir)
        pickle.dump(feature_selection, 
                    open(os.path.join(model_dir, f'{layer}_feat_model_{args.num_feat}{args.adapt_to}.pkl'), 
                        'wb'))

# =============================================================================
# Save new features
# =============================================================================

np.save(os.path.join(save_dir, 'dnn_feature_maps', 
                     f'new_feature_maps_{args.num_feat}{args.adapt_to}'), best_feat) 