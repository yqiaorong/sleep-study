import os
import scipy
import numpy as np
from tqdm import tqdm
from func import mvnn
import argparse
import mat73
import pickle
from scipy import stats

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=2, type=int)
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

eeg_fname = f'sub-{args.sub:03d}_task-localiser_source_data'
eeg_data = mat73.loadmat(os.path.join(eeg_dir, eeg_fname+'.mat'))
eeg = eeg_data['sub_eeg_loc']['eeg']
print('Original eeg_data shape (trial, ch, time)', eeg.shape)

eeg_labels = eeg_data['sub_eeg_loc']['images']

# set channels
num_vox = eeg.shape[1]

# set time
t_sleemory = eeg.shape[2]

# =============================================================================
# Categorize the preprocessed data
# =============================================================================

# Find indices in A that match the first element of B
eeg_labels  = [s[0].split('-')[0] for s in eeg_labels] # no .jpg and idx
eeg_labels = np.asarray(eeg_labels)
print(eeg_labels)
unique_imgs = np.unique(eeg_labels)
print(unique_imgs.shape)

# Sort the test eeg data
tot_img_indices = []
tot_eeg = [] # storing all EEG for each category
# Iterate over images
for idx, img in enumerate(tqdm(unique_imgs, desc='unique images')):
    img_indices = np.where(eeg_labels == img)[0]
    print(img_indices)
    tot_img_indices.append(img_indices)
    
    print(img)
    print(eeg_labels[img_indices])

    # select corresponding prepr data
    select_eeg = eeg[img_indices]
    print(select_eeg.shape)

    tot_eeg.append(select_eeg)

# =============================================================================
# Whitening 
# =============================================================================

whiten_data = mvnn(tot_eeg)
del tot_eeg

# Assign the whitened data back in original order
whitened_data_re = np.empty(eeg.shape) 
for indices, data in zip(tot_img_indices, whiten_data):
    whitened_data_re[indices] = data
print(f'Total eeg_data shape after whiten (img, ch, time)', whitened_data_re.shape)

# # Average z scored total test eeg data
# whitened_data_unique = np.empty((len(unique_imgs), num_vox, t_sleemory))
# for i, data in enumerate(whiten_data):
#     new_data = np.mean(data, axis=0)
#     whitened_data_unique[i] = new_data
# print(f'Unique eeg_data shape after whiten ({args.whiten}) (img, ch, time)', whitened_data_unique.shape)

del whiten_data

# =============================================================================
# Z score the data
# =============================================================================

# zscored_data = np.empty(eeg.shape)
# if args.whiten == True:
#     # tot_test_eeg = mvnn(tot_test_eeg)
#     for t in tqdm(range(t_sleemory), desc='time'):
#         for vox in range(num_vox):
#             zscored_data[:, vox, t] = stats.zscore(eeg[:,vox,t], axis = 0) 

# =============================================================================
# Save the test eeg data
# =============================================================================

# Create the saving directory
save_dir = f'output/sleemory_localiser_vox/whiten_eeg'
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)

# np.save(os.path.join(save_dir, eeg_fname), {'eeg': whitened_data_re, 
#                                             'images': eeg_labels})
scipy.io.savemat(os.path.join(save_dir, eeg_fname), {'eeg': whitened_data_re, 
                                                     'images': eeg_labels})