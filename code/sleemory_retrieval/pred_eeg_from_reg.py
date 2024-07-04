import os
import pickle
import scipy
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sleemory_retrieval', type=str)
parser.add_argument('--z_score', default=True, type=bool)
parser.add_argument('--num_feat', default=1000, type=str)
args = parser.parse_args()

print('')
print(f'>>> Apply regression model on sleemory retrieval <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')



# Save directory
save_dir = f'output/{args.dataset}/test_pred_eeg'
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)
    
    

# =============================================================================
# Load the data
# =============================================================================
        
# Load fmaps
fmaps_path = f'dataset/{args.dataset}/dnn_feature_maps/best_fmaps/new_feature_maps_{args.num_feat}.npy'
fmaps = np.load(fmaps_path, allow_pickle=True).item()

save_eeg = {}
for key, value in fmaps.items():
    print(f'The layer {key} has fmaps shape (img, feat) {fmaps[key].shape}')
    print('')

    # =============================================================================
    # Apply the model
    # =============================================================================

    # Apply the encoding model
    reg = pickle.load(open(os.path.join('dataset/sleemory_localiser/model/reg_model', 
                                        f'{key}_reg_model.pkl'), 'rb'))

    # Predict EEG
    pred_eeg = reg.predict(value)

    # Reshape the test data and the predicted data
    pred_eeg = pred_eeg.reshape(value.shape[0], 58, 363)
    print('Predicted EEG data shape (img, ch x time)', pred_eeg.shape)
    
    save_eeg[key] = pred_eeg

np.save(os.path.join(save_dir, f'pred_eeg_with_{args.num_feat}feats'), save_eeg)
scipy.io.savemat(os.path.join(save_dir, f'pred_eeg_with_{args.num_feat}feats.mat'), save_eeg) 