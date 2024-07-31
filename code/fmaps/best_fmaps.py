import os
import pickle
import argparse
import numpy as np
import scipy
from sklearn.feature_selection import SelectKBest, f_regression

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--networks', default=None, type=str) # [gptneo / resnext]
parser.add_argument('--num_feat', default=1000, type=int)
parser.add_argument('--dataset',  default=None, type=str)
parser.add_argument('--whiten',   default=False,type=bool)
args = parser.parse_args()

print('')  
print('Feature selection of sleemory images feature maps <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')

# Set up
if args.whiten == False:
    whiten = ''
else:
    whiten = 'whiten_'

# =============================================================================
# Load the feature maps
# =============================================================================

fmaps = scipy.io.loadmat(f'dataset/sleemory_{args.dataset}/dnn_feature_maps/{args.networks}_fmaps.mat')
fmaps = fmaps['fmaps']
print('The original fmaps shape: ', fmaps.shape)

# =============================================================================
# Feature selection
# =============================================================================

# Set up the model dir and name
model_fname = f'{args.networks}-best-{args.num_feat}_{whiten}feat_model.pkl'
print(model_fname)

# Fit feature selection model and transform
if args.num_feat == -1:
    best_feat = fmaps
else:  
    if args.dataset == 'localiser':
        # Load eeg data
        data_fname = f'unique_{whiten}eeg.mat'
        print(data_fname)
        data = scipy.io.loadmat(f'output/sleemory_{args.dataset}/whiten_eeg/{data_fname}')
        eeg_data = data[f'{whiten}eeg']  # (num_img, num_ch, num_time)
        # Average across time
        eeg_data = np.mean(eeg_data, -1) # (num_img, num_ch,)
        # Average across channel
        eeg_data = np.mean(eeg_data, -1) # (num_img,)
        print('EEG data shape (img,)', eeg_data.shape)
        
        # Build the feature selection model
        feature_selection = SelectKBest(f_regression, 
                                        k=args.num_feat).fit(fmaps, eeg_data)
        best_feat = feature_selection.transform(fmaps)
        
        # Save the model
        model_dir = f'dataset/sleemory_'+args.dataset+'/model/best_fmaps_model'
        if os.path.isdir(model_dir) == False:
            os.makedirs(model_dir)
        pickle.dump(feature_selection, open(f'{model_dir}/{model_fname}', 'wb'))
        del feature_selection
        
    elif args.dataset == 'retrieval':
        # Apply the model
        with open(f'dataset/sleemory_localiser/model/best_fmaps_model/{model_fname}', 'rb') as f:
            feature_selection = pickle.load(f)
            best_feat = feature_selection.transform(fmaps)
        
    print(f'The new fmaps shape: {best_feat.shape}')
del fmaps

# =============================================================================
# Save new features
# =============================================================================

save_dir = f'dataset/sleemory_{args.dataset}/dnn_feature_maps'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
 
# Save new features
best_fmaps_fname = f'{args.networks}-best-{args.num_feat}_{whiten}fmaps.mat'
print(best_fmaps_fname)
scipy.io.savemat(f'{save_dir}/{best_fmaps_fname}', {'fmaps': best_feat}) 