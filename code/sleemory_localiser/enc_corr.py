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
parser.add_argument('--dataset', default='sleemory_localiser', type=str)
parser.add_argument('--z_score', default=True,  type=bool)
parser.add_argument('--num_feat', default=1000, type=str)
parser.add_argument('--layer_name', default='conv5', type=str)
args = parser.parse_args()

print('')
print(f'>>> Encoding model within sleemory <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')

# =============================================================================
# Load the data
# =============================================================================

# Load whitened eeg data
eeg_path = f'output/{args.dataset}/whiten_eeg/whiten_test_eeg.npy'
load_eeg = np.load(eeg_path, allow_pickle=True).item()
eeg = load_eeg['test_eeg2']

# Get the parameters
num_stimuli = eeg.shape[0]
num_ch = eeg.shape[1]
num_time = eeg.shape[2]

# Reshape the eeg data
eeg = eeg.reshape(eeg.shape[0], -1)
print('EEG data shape (img, ch x time)', eeg.shape)

# Load fmaps
fmaps_path = f'dataset/{args.dataset}/dnn_feature_maps/best_fmaps/new_feature_maps_{args.num_feat}.npy'
fmaps = np.load(fmaps_path, allow_pickle=True).item()
print(f'The layer {args.layer_name} has fmaps shape (img, feat) {fmaps[args.layer_name].shape}')

# =============================================================================
# Split the data
# =============================================================================

# Split the data into training and test partitions
rand_seed = 5 #@param
np.random.seed(rand_seed)

# Get the number of training and test data
num_train = int(np.round(num_stimuli / 100 * 75))
num_test = num_stimuli - num_train

# Shuffle all data
idxs = np.arange(num_stimuli)
np.random.shuffle(idxs)

# Get training and test data
idxs_train, idxs_test = idxs[:num_train], idxs[num_train:]

# Training data
train_eeg, test_eeg = eeg[idxs_train], eeg[idxs_test]
print(f'The training partition eeg data shape: {train_eeg.shape}')
print(f'The test partition eeg data shape: {test_eeg.shape}')
del eeg

# Test data
train_fmaps, test_fmaps = fmaps[args.layer_name][idxs_train], fmaps[args.layer_name][idxs_test]
print(f'The training partition fmaps shape: {train_fmaps.shape}')
print(f'The test partition fmaps shape: {test_fmaps.shape}')
del fmaps

# =============================================================================
# Encoding
# =============================================================================

# Save directory
save_dir = f'output/{args.dataset}'
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)

# Train the encoding model
reg_dir = f'dataset/{args.dataset}/model/reg_model'
if os.path.isdir(reg_dir) == False:
    os.makedirs(reg_dir)
        
if os.path.isdir(os.path.join(reg_dir, f'{args.layer_name}_reg_model.pkl')) == False:
    reg = LinearRegression().fit(train_fmaps, train_eeg)
    pickle.dump(reg, open(os.path.join(reg_dir, f'{args.layer_name}_reg_model.pkl'), 'wb'))
else:
    reg = pickle.load(open(os.path.join(reg_dir, f'{args.layer_name}_reg_model.pkl'), 'rb'))

pred_eeg = reg.predict(test_fmaps)

# =============================================================================
# Svae predicted EEG
# =============================================================================

save_data_dir = os.path.join(save_dir, 'test_pred_eeg', f'{args.num_feat}feats')
if os.path.isdir(save_data_dir) == False:
	os.makedirs(save_data_dir)

# Reshape the test data and the predicted data
train_eeg = train_eeg.reshape(num_train, num_ch, num_time)
pred_eeg = pred_eeg.reshape(num_test, num_ch, num_time)
test_eeg = test_eeg.reshape(num_test, num_ch, num_time)
print('Predicted EEG data shape (img, ch x time)', pred_eeg.shape)

# Save the predicted and real eeg data
save_eeg = {'train_eeg': train_eeg, 'test_eeg': test_eeg, 'pred_eeg': pred_eeg}

np.save(os.path.join(save_data_dir, f'{args.layer_name}_eeg'), save_eeg)
scipy.io.savemat(os.path.join(save_data_dir, f'{args.layer_name}_eeg.mat'), save_eeg) 

# =============================================================================
# Correlation
# =============================================================================

enc_dir = os.path.join(save_dir, f'enc_acc/{args.num_feat}feats')
if os.path.isdir(enc_dir) == False:
    os.makedirs(enc_dir)
    
enc_acc = np.empty((num_test, num_time, num_time))
            
# Correlation across stimuli
for stimuli_idx in tqdm(range(num_test), desc='Iteration over image'):
    for t_test in range(num_time):
        test_val = pd.Series(test_eeg[stimuli_idx, :, t_test])
        for t_pred in range(num_time):
            pred_val = pd.Series(pred_eeg[stimuli_idx, :, t_pred])
            enc_acc[stimuli_idx, t_test, t_pred] = test_val.corr(pred_val)

# Average the results over stimuli
avg_enc_acc = np.mean(enc_acc, axis=0)

# Save data
np.save(os.path.join(enc_dir, f'{args.layer_name}_enc_acc'), enc_acc)
scipy.io.savemat(os.path.join(enc_dir, f'{args.layer_name}_enc_acc.mat'), enc_acc) 
del enc_acc

# =============================================================================
# Plot the correlation results
# =============================================================================

# Plot all 2D results of method 1
fig = plt.figure(figsize=(6, 5))
im = plt.imshow(avg_enc_acc, cmap='viridis',
				extent=[-0.25, 1, -0.25, 1], 
                origin='lower', aspect='auto')
cbar = plt.colorbar(im)
cbar.set_label('Values')

# Plot borders
plt.plot([-0.25, 1], [0,0], 'k--', lw=0.4)
plt.plot([0,0], [-0.25, 1], 'k--', lw=0.4)
plt.xlim([-0.25, 1])
plt.ylim([-0.25, 1])
plt.xlabel('Test EEG time / s')
plt.ylabel('Pred EEG time / s')
plt.title(f'Encoding accuracies of layer {args.layer_name}')
fig.tight_layout()
plt.savefig(os.path.join(enc_dir, f'{args.layer_name}_enc_acc'))