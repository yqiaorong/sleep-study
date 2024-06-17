import os
import scipy
import numpy as np
from tqdm import tqdm
from func import mvnn
import argparse

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--adapt_to', default=None, type=str)
args = parser.parse_args()

print('')
print(f'>>> Whiten EEG data <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')

# =============================================================================
# Predict EEG from fmaps
# =============================================================================

# Load the test EEG data directory
eeg_dir = 'dataset/sleemory_localiser/preprocessed_data'
data = scipy.io.loadmat(os.path.join(eeg_dir,'sleemory_localiser_dataset.mat'))
prepr_data = data['ERP_all']
imgs_all = data['imgs_all']
print('Original test_eeg_data shape (img, ch, time)', prepr_data.shape)

# set channels
ch_names = []
for ch in data['channel_names'][:,0]:
    ch_names.append(ch[0])

if args.adapt_to == '_THINGS':
    # Drop the extra channel 'Fpz' and 'Fz':
    idx_Fz, idx_Fpz = ch_names.index('Fz'), ch_names.index('Fpz')
    prepr_data = np.delete(prepr_data, [idx_Fz, idx_Fpz], axis=1)
    ch_names = np.delete(ch_names, [idx_Fz, idx_Fpz])

num_ch = len(ch_names)
print('Final test_eeg_data shape (img, ch, time)', prepr_data.shape)

# set time
times = data['time']
t_sleemory = times.shape[1]
del data



# =============================================================================
# Categorize the preprocessed data
# =============================================================================

# Find indices in A that match the first element of B
unique_imgs = np.unique(imgs_all)
unique_imgs = [item for img in unique_imgs for item in img]

# Sort the test eeg data
tot_test_eeg = [] # storing all EEG for each img
# Iterate over images
for idx, img in enumerate(tqdm(unique_imgs, desc='Average test eeg across unique images')):
    img_indices = np.where(imgs_all == img)[0]
    # select corresponding prepr data
    select_data = prepr_data[img_indices]
    # Append data
    tot_test_eeg.append(select_data)
    
    

# =============================================================================
# Z score the data
# =============================================================================

tot_test_eeg = mvnn(tot_test_eeg)

# Average z scored total test eeg data
test_eeg2 = np.empty((len(unique_imgs), num_ch, t_sleemory))
for i, data in enumerate(tot_test_eeg):
    new_data = np.mean(data, axis=0)
    test_eeg2[i] = new_data
del tot_test_eeg



# =============================================================================
# Save the test eeg data
# =============================================================================

# Create the saving directory
save_dir = f'output/sleemory_localiser{args.adapt_to}/whiten_eeg'
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)
    
save_dict = {'test_eeg2': test_eeg2}
np.save(os.path.join(save_dir, f'whiten_test_eeg'), save_dict)
scipy.io.savemat(os.path.join(save_dir, f'whiten_test_eeg.mat'), save_dict)