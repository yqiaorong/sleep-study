import os
import pickle
import argparse
import numpy as np
import scipy
from func import load_sleemory_full_fmaps
from sklearn.feature_selection import SelectKBest, f_regression

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--num_feat',   default=1000, type=int)
parser.add_argument('--dataset',    default=None, type=str)
args = parser.parse_args()

print('')
print('Feature selection of sleemory images feature maps <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')

# =============================================================================
# Load the feature maps
# =============================================================================

fmaps = load_sleemory_full_fmaps(args)
print('')

# Drop the extra fmaps
if args.dataset == 'sleemory_localiser':
    drop_idx = 0
    for key, value in fmaps.items():
        fmaps[key] = np.delete(fmaps[key], drop_idx, axis=0)
    print(f'The layer {key} has final fmaps shape (img, feat) {fmaps[key].shape}')

# =============================================================================
# Feature selection
# =============================================================================

# Fit feature selection model and transform
if args.num_feat == -1:
    best_feat = fmaps
else:
    best_feat = {}
    # layers names
    all_layers = fmaps.keys()
    for layer in all_layers:
        
        if args.dataset == 'sleemory_localiser':
            # Load eeg data
            eeg_path = f'output/{args.dataset}/test_eeg/whiten_test_eeg.npy'
            data = np.load(eeg_path, allow_pickle=True).item()
            eeg_data = data['test_eeg2']
            # Average across time to shape (img, ch,)
            eeg_data = np.mean(eeg_data, -1)
            # Average across channel to shape (img,)
            eeg_data = np.mean(eeg_data, -1)
            print('EEG data shape (img,)', eeg_data.shape)
            
            # Build the feature selection model
            feature_selection = SelectKBest(f_regression, 
                                            k=args.num_feat).fit(fmaps[layer], eeg_data)
            best_feat[layer] = feature_selection.transform(fmaps[layer])
            
            # Save the model
            model_dir = f'dataset/'+args.dataset+'/model/best_fmaps_model'
            if os.path.isdir(model_dir) == False:
                os.makedirs(model_dir)
            pickle.dump(feature_selection, 
                        open(os.path.join(model_dir, f'{layer}_feat_model_{args.num_feat}.pkl'), 
                            'wb'))
            
        elif args.dataset == 'sleemory_retrieval':
            # Apply the model
            feat_model_dir = os.path.join('dataset/sleemory_localiser/model/best_fmaps_model',
                                  f'{layer}_feat_model_{args.num_feat}.pkl')
            with open(feat_model_dir, 'rb') as f:
                feature_selection = pickle.load(f)
                best_feat[layer] = feature_selection.transform(fmaps[layer])
                
        print(f'The new fmaps of layer {layer} has shape {best_feat[layer].shape}')
del fmaps

# =============================================================================
# Save new features
# =============================================================================

save_dir = os.path.join('dataset', args.dataset, 'dnn_feature_maps', 'best_fmaps')
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
 
# Save new features
np.save(os.path.join(save_dir, f'new_feature_maps_{args.num_feat}'), best_feat) 
scipy.io.savemat(os.path.join(save_dir, f'new_feature_maps_{args.num_feat}.mat'), best_feat) 