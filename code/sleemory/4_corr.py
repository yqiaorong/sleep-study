import os
import scipy
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from func import mvnn, mvnn_mean
from matplotlib import pyplot as plt
from scipy.stats import pearsonr as corr

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--layer_name', default='conv5', type=str)
parser.add_argument('--num_feat', default=1000, type=int)
parser.add_argument('--enc_img_range', default=[0, 100], nargs='+', type=float)
args = parser.parse_args()

print('')
print(f'>>> Test the encoding model on sleemory (pattern) <<<')
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
for f, fmaps in enumerate(tqdm(fmaps_list, desc='load sleemory images')):
    fmaps_data = np.load(os.path.join(fmaps_dir, fmaps),
                            allow_pickle=True).item()
    all_layers = fmaps_data.keys()
    for l, dnn_layer in enumerate(all_layers):
        if f == 0:
            feats.append([[np.reshape(fmaps_data[dnn_layer], -1)]])
        else:
            feats[l].append([np.reshape(fmaps_data[dnn_layer], -1)])
    
fmaps_test[args.layer_name] = np.squeeze(np.asarray(feats[l]))
print(f'The original fmaps shape (img, feat)', fmaps_test[args.layer_name].shape)

# =============================================================================
# Load the feature selection model and apply feature selection to test fmaps
# =============================================================================

model_path = 'dataset/temp_sleemory'
feat = pickle.load(open(os.path.join(model_path, 
                                     f'feat_model_{args.num_feat}.pkl'), 'rb'))
best_feat_test = feat.transform(fmaps_test[args.layer_name])
print(f'The new fmaps shape (img, feat) {best_feat_test.shape}')

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
# set time
t_sleemory, t_THINGS = times.shape[1], 250
del data
# Drop the extra channel 'Fpz' and 'Fz':
idx_Fz, idx_Fpz = ch_names.index('Fz'), ch_names.index('Fpz')
prepr_data = np.delete(prepr_data, [idx_Fz, idx_Fpz], axis=1)
print('Original test_eeg_data shape (img, ch, time)', prepr_data.shape)

# set channels
num_ch = len(ch_names)-2

# Predict the EEG data 
pred_eeg = reg.predict(best_feat_test)
# Reshape the predicted EEG data
pred_eeg = np.reshape(pred_eeg, (pred_eeg.shape[0], num_ch, t_THINGS))
print('pred_eeg_data shape (img, ch, time)', pred_eeg.shape)

# =============================================================================
# Categorize the preprocessed data
# =============================================================================

# Find indices in A that match the first element of B
unique_imgs = np.unique(imgs_all)
unique_imgs = [item for img in unique_imgs for item in img]

image_set_list = os.listdir('dataset/temp_sleemory/image_set')

# There is one extra fmap which should be dropped from pred eeg
exclude_img = list(set(image_set_list)-set(unique_imgs))
exclude_idx_in_img_set = image_set_list.index(exclude_img[0])
pred_eeg = np.delete(pred_eeg, exclude_idx_in_img_set, axis=0)
print("One img is dropped from pred eeg since it's absent in test eeg.")

# Sort the test eeg data
test_eeg = np.empty((len(unique_imgs), num_ch, t_sleemory)) # storing mean EEG for each img
tot_test_eeg = [] # storing all EEG for each img
# Iterate over images
for idx, img in enumerate(tqdm(unique_imgs, desc='Average test eeg across unique images')):
    img_indices = np.where(imgs_all == img)[0]
    # select corresponding prepr data
    select_data = prepr_data[img_indices]
    # Append data
    tot_test_eeg.append(select_data)
    
    # Average across the same images
    select_data = np.mean(select_data, 0)
    test_eeg[idx] = select_data
    
# =============================================================================
# Z score the data
# =============================================================================

if args.z_score == True:
    test_eeg = mvnn_mean(test_eeg)
    tot_test_eeg = mvnn(tot_test_eeg)
else:
    pass

# Average z scored total test eeg data
test_eeg2 = np.empty(test_eeg.shape)
for i, data in enumerate(tot_test_eeg):
    new_data = np.mean(data, axis=0)
    test_eeg2[i] = new_data
del tot_test_eeg

# =============================================================================
# Test the encoding model
# =============================================================================

print('Final pred_eeg_data shape (img, ch, time)', pred_eeg.shape)
print('Final test_eeg_data shape (img, ch, time)', test_eeg.shape, test_eeg2.shape)

img_cond_idx = 0

# Calculate the encoding accuracy
enc_acc = np.empty((t_THINGS, t_sleemory))
enc_acc2 = np.empty((t_THINGS, t_sleemory))
for t_s in range(t_sleemory):
    for t_TH in range(t_THINGS):
        enc_acc[t_TH, t_s] = corr(pred_eeg[img_cond_idx, :, t_TH],
                                    data[img_cond_idx, :, t_s])[0]
        enc_acc2[t_TH, t_s] = corr(pred_eeg[img_cond_idx, :, t_TH],
                                    data[img_cond_idx, :, t_s])[0]
        
# Create the saving directory
save_dir = 'output/sleemory/enc_acc'
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)
    
# # Save the results
# with open(os.path.join(save_dir, f'enc_acc_{args.enc_img_range[0]}'), 
#           'wb') as f: 
#     pickle.dump(enc_acc, f, protocol=4) 

# =============================================================================
# Plot the correlation results
# =============================================================================
    
# Plot all 2D results of method 1
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
plt.title(f'Encoding accuracy (pattern) with {args.num_feat} features')
fig.tight_layout()
plt.savefig(os.path.join(save_dir, f'enc acc (pattern) M1 ({args.num_feat} feats) z scored {args.z_score}'))

# Plot all 2D results of method 2
fig = plt.figure(figsize=(6, 5))
im = plt.imshow(enc_acc2, cmap='viridis',
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
plt.title(f'Encoding accuracy (pattern) with {args.num_feat} features')
fig.tight_layout()
plt.savefig(os.path.join(save_dir, 
                         f'enc acc (pattern) M2 ({args.num_feat} feats) z scored {args.z_score} '))