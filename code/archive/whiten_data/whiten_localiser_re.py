"""This script whitens the sleemory localiser eeg and keeps the original order 
by stacking eegs with the same stimuli, whitening them and then assigning them
back according to original stimuli order. Not quite useful for later analysis.
"""

import os
import scipy
import numpy as np
from tqdm import tqdm
from func import mvnn

print('')
print(f'>>> Whiten sleemory localiser EEG data <<<')
print('')

# =============================================================================
# Predict EEG from fmaps
# =============================================================================

# Load the test EEG data directory
eeg_dir = 'dataset/sleemory_localiser/preprocessed_data'
data = scipy.io.loadmat(os.path.join(eeg_dir,'sleemory_localiser_dataset.mat'))
prepr_data = data['ERP_all'] # (img, ch, time)
imgs_all = data['imgs_all']
print('eeg_data shape (img, ch, time)', prepr_data.shape)



# =============================================================================
# Categorize the preprocessed data and whiten the data
# =============================================================================

# Find indices in A that match the first element of B
unique_imgs = np.unique(imgs_all)
unique_imgs = [item for img in unique_imgs for item in img]


# Iterate over images
whitened_data_re = np.empty(prepr_data.shape) # (img, ch, time)
for name in tqdm(unique_imgs, desc='unique images'):
    true_idx = np.where(imgs_all == name)[0]

    # whiten data
    whiten_data = mvnn([prepr_data[true_idx]])
    whiten_data = np.squeeze(np.asarray(whiten_data)) # (same_img, ch, time)

    # Assign the whitened data to final whitened data with original order
    whitened_data_re[true_idx] = whiten_data
    del whiten_data



# =============================================================================
# Save the whitened eeg data
# =============================================================================

# Create the saving directory
save_dir = f'output/sleemory_localiser/whiten_eeg'
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)

scipy.io.savemat(os.path.join(save_dir, f'origin_whiten_eeg.mat'), 
                 {'whiten_eeg': whitened_data_re})