import os
import scipy
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.stats import pearsonr as corr

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--layer_name', default='conv5', type=str)
parser.add_argument('--num_feat', default=300, type=int)
args = parser.parse_args()

print('')
print(f'>>> Test the encoding model on sleemory <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
 
# =============================================================================
# Load the full test feature maps
# =============================================================================

feats = []
fmaps_test = {}
# The directory of the dnn training feature maps
fmaps_dir = os.path.join('dataset','temp_sleemory','dnn_feature_maps',
                        'full_feature_maps', 'alexnet', 
                        'pretrained-'+str(args.pretrained))
fmaps_list = os.listdir(fmaps_dir)
fmaps_list.sort()
for f, fmaps in enumerate(tqdm(fmaps_list, desc='load sleemory training images')):
    fmaps_data = np.load(os.path.join(fmaps_dir, fmaps),
                            allow_pickle=True).item()
    all_layers = fmaps_data.keys()
    for l, dnn_layer in enumerate(all_layers):
        if f == 0:
            feats.append([[np.reshape(fmaps_data[dnn_layer], -1)]])
        else:
            feats[l].append([np.reshape(fmaps_data[dnn_layer], -1)])
    
fmaps_test[args.layer_name] = np.squeeze(np.asarray(feats[l]))
print(f'The original fmaps shape', fmaps_test[args.layer_name].shape)

# =============================================================================
# Load the feature selection model and apply feature selection to test fmaps
# =============================================================================

model_path = 'dataset/temp_sleemory'
feat = pickle.load(open(os.path.join(model_path, 
                                     f'feat_model_{args.num_feat}.pkl'), 'rb'))
best_feat_test = feat.transform(fmaps_test[args.layer_name])
print(f'The new fmaps shape {best_feat_test.shape}')

# =============================================================================
# Load the encoding model
# =============================================================================

reg = pickle.load(open(os.path.join(model_path, 
                                    f'reg_model_{args.num_feat}.pkl'), 'rb'))

# =============================================================================
# Load the test EEG data and predict EEG from fmaps
# =============================================================================

# Load the test EEG data directory
eeg_dir = os.path.join('dataset', 'temp_sleemory', 'preprocessed_data')
data = scipy.io.loadmat(os.path.join(eeg_dir,'sleemory_localiser_dataset.mat'))
prepr_data = data['ERP_all']
imgs_all = data['imgs_all']
ch_names = []
for ch in data['channel_names'][:,0]:
    ch_names.append(ch[0])
times = data['time']
del data
# Drop the extra channel 'Fpz' and 'Fz':
idx_Fz, idx_Fpz = ch_names.index('Fz'), ch_names.index('Fpz')
prepr_data = np.delete(prepr_data, [idx_Fz, idx_Fpz], axis=1)
# # Average the training EEG data over time: (num_feat,num_ch)
# prepr_data = np.mean(prepr_data, -1)
# # Average the training EEG data across electrodes: (num_feat,)
# prepr_data = np.mean(prepr_data, -1)

# Predict the EEG data 
pred_eeg = reg.predict(best_feat_test)
# Reshape the predicted EEG data
pred_eeg = np.reshape(pred_eeg, (pred_eeg.shape[0],56,250))
print('pred_eeg_data shape', pred_eeg.shape)

# =============================================================================
# Categorize the preprocessed data
# =============================================================================

# Find indices in A that match the first element of B
unique_imgs = np.unique(imgs_all)

### Test the encoding model ###
enc_acc = np.empty((len(unique_imgs), 250, times.shape[1]))
    
# Iterate over images
for i in tqdm(range(len(unique_imgs)), desc='sleemory images'):
    img_indices = np.where(imgs_all == unique_imgs[i])[0]
    # select corresponding prepr data
    select_data = prepr_data[img_indices]
    # Average across the same images
    select_data = np.mean(select_data, 0)

    # Calculate the encoding accuracy
    for t_sleep in range(times.shape[1]):
        for t_THINGS in range(250):
            enc_acc[i, t_THINGS, t_sleep] = corr(pred_eeg[i,:,t_THINGS],
                select_data[:,t_sleep])[0]
# Average the encoding accuracy across images
enc_acc = np.mean(enc_acc, 0)
print(enc_acc.shape)

# =============================================================================
# Plot the correlation results
# =============================================================================
    
# Create the saving directory
save_dir = 'output/sleemory'
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)

# Plot all 2D results
fig = plt.figure(figsize=(6, 5))
im = plt.imshow(enc_acc, cmap='viridis',
				extent=[-0.2, 0.8, -0.25, 1], 
                origin='lower', aspect='auto')
cbar = plt.colorbar(im)
cbar.set_label('Values')
# Plot borders
plt.plot([-0.2, 0.8], [0,0], 'k--', lw=0.4)
plt.plot([0,0], [-0.25, 1], 'k--', lw=0.4)

plt.xlim([-0.2, 0.8])
plt.ylim([-0.25, 1])
plt.xlabel('THINGS time / s')
plt.ylabel('Sleemory time / s')
plt.title(f'Encoding accuracy')
fig.tight_layout()
plt.savefig(os.path.join(save_dir, f'encoding accuracy'))

# # Plot the diagonal
# fig = plt.figure(figsize=(6, 3))
# # Select the diagonal elements
# diag_enc_acc = []
# for t in range(250):
#     diag_enc_acc.append(enc_acc[t,t])
# plt.plot(np.linspace(-0.2, 0.8, 250), diag_enc_acc)
# # Plot borders
# plt.plot([-0.2, 0.8], [0,0], 'k--', lw=0.4)
# plt.plot([0,0], [-0.25, 1], 'k--', lw=0.4)
# plt.xlabel('THINGS time / s')
# plt.ylabel('Accuracy')
# plt.title(f'Diagonal encoding accuracy')
# fig.tight_layout()
# plt.savefig(os.path.join(save_dir, f'diagonal encoding accuracy'))