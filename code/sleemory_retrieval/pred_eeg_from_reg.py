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

# # =============================================================================
# # Correlation
# # =============================================================================

# enc_acc = np.empty((num_test, num_time, num_time))
            
# # Correlation across stimuli
# for stimuli_idx in tqdm(range(num_test), desc='Iteration over stimuli'):
#     for t_test in range(num_time):
#         test_val = pd.Series(test_eeg[stimuli_idx, :, t_test])
#         for t_pred in range(num_time):
#             pred_val = pd.Series(pred_eeg[stimuli_idx, :, t_pred])
#             enc_acc[stimuli_idx, t_test, t_pred] = test_val.corr(pred_val)

# # Average the results over stimuli
# avg_enc_acc = np.mean(enc_acc, axis=0)

# # Save data
# np.save(os.path.join(save_dir, f'{args.layer}_enc_acc'), enc_acc)
# del enc_acc

# # =============================================================================
# # Plot the correlation results
# # =============================================================================

# # Plot all 2D results of method 1
# fig = plt.figure(figsize=(6, 5))
# im = plt.imshow(avg_enc_acc, cmap='viridis',
# 				extent=[-0.25, 1, -0.25, 1], 
#                 origin='lower', aspect='auto')
# cbar = plt.colorbar(im)
# cbar.set_label('Values')

# # Plot borders
# plt.plot([-0.25, 1], [0,0], 'k--', lw=0.4)
# plt.plot([0,0], [-0.25, 1], 'k--', lw=0.4)
# plt.xlim([-0.25, 1])
# plt.ylim([-0.25, 1])
# plt.xlabel('Test EEG time / s')
# plt.ylabel('Pred EEG time / s')
# plt.title('Encoding accuracies')
# fig.tight_layout()
# plt.savefig(os.path.join(save_dir, f'{args.layer}_enc_acc'))