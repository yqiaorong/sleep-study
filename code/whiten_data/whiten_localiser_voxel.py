import os
import scipy
import numpy as np
from tqdm import tqdm
# from func import mvnn
import argparse
import mat73
import pickle
from scipy import stats

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=2, type=int)
parser.add_argument('--whiten', default=False, type=bool)
args = parser.parse_args()

print('')
print(f'>>> Whiten EEG data (voxel) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')

# =============================================================================
# Load raw localiser EEG data
# =============================================================================

# Load the test EEG data directory
eeg_dir = '/home/simon/Documents/gitrepos/shannon_encodingmodelsEEG/dataset/sleemory_localiser/preprocessed_data'

fname = f'sub-{args.sub:03d}_task-localiser_source_data'
data = mat73.loadmat(os.path.join(eeg_dir, fname+'.mat'))
prepr_data = data['sub_eeg_loc']['eeg']

imgs_all = data['sub_eeg_loc']['images']
imgs_all = [s[0].split('.')[0] for s in imgs_all]
imgs_all = np.asarray(imgs_all)

print('Original eeg_data shape (trial, ch, time)', prepr_data.shape)

# set channels
num_vox = prepr_data.shape[1]

# set time
t_sleemory = prepr_data.shape[2]

# =============================================================================
# Categorize the preprocessed data
# =============================================================================

# # Find indices in A that match the first element of B
# imgs_all = [s[0].split('-')[0] for s in imgs_all]
# imgs_all = np.asarray(imgs_all)
# unique_imgs = np.unique(imgs_all)

# # Sort the test eeg data
# tot_img_indices = []
# tot_test_eeg = [] # storing all EEG for each img
# # Iterate over images
# for idx, img in enumerate(tqdm(unique_imgs, desc='unique images')):
#     img_indices = np.where(imgs_all == img)[0]
#     tot_img_indices.append(img_indices)

#     # select corresponding prepr data
#     select_data = prepr_data[img_indices]
#    stats.zscore(tot_test_eeg[:,:,t], axis =0) 
#     # standardize
# Find indices in A that match the first element of B
# imgs_all = [s[0].split('-')[0] for s in imgs_all]
# imgs_all = np.asarray(imgs_all)
# unique_imgs = np.unique(imgs_all)
#     # Append data
#     tot_test_eeg.append(select_data)
#     print(select_data.shape)

# Z score the data
zscored_data = np.empty(prepr_data.shape)
if args.whiten == True:
    # tot_test_eeg = mvnn(tot_test_eeg)
    for t in tqdm(range(t_sleemory), desc='time'):
        for vox in range(num_vox):
            zscored_data[:, vox, t] =  stats.zscore(prepr_data[:,vox,t], axis = 0) 

# # Assign the whitened data back in original order
# whitened_data_re = np.empty(prepr_data.shape) 
# for indices, test_eeg in zip(tot_img_indices, tot_test_eeg):
#     whitened_data_re[indices] = test_eeg
# print(f'Total eeg_data shape after whiten ({args.whiten}) (img, ch, time)', whitened_data_re.shape)

# # Average z scored total test eeg data
# test_eeg = np.empty((len(unique_imgs), num_vox, t_sleemory))
# for i, data in enumerate(tot_test_eeg):
#     new_data = np.mean(data, axis=0)
#     test_eeg[i] = new_data
# print(f'Unique eeg_data shape after whiten ({args.whiten}) (img, ch, time)', test_eeg.shape)

# del tot_test_eeg

# =============================================================================
# Save the test eeg data
# =============================================================================

# Create the saving directory
save_dir = f'output/sleemory_localiser_vox/whiten_eeg'
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)

if args.whiten == True:
    fname_insert = 'whiten_'
else:
    fname_insert = ''
np.save(os.path.join(save_dir, fname), {#f'{fname_insert}eeg_unique': test_eeg, 
#                                                 'unique_img': unique_imgs,
                                                f'{fname_insert}eeg': zscored_data, 
                                                'imgs_all': imgs_all})
pickle.dump