"""INSTRUCTIONS
Preprocess the THINGS EEG2 data (change the sample frequency to 259Hz and 
re-check the electrodes.)

Sleemory has preprocessed data with shape: 
trials x electrodes x times (18k x 58 x 365)

Use Alexnet ('convo5') to extract the feature maps of images

Select the best 100 features using the feature_selection model from THINGS EEG2

Input the new feature maps of sleemory to the encoding model and output the
predicted sleemory EEG data with shape (58, 100)

The correlation between the predicted sleemory EEG data (58 x 100) and the real
sleemory EEG data (58 x 365): do a for loop:
for i in pred[:,]:
    for j in real[:,]:
        corr(i,j)
Then plot a 2d matrix of accuracies. The horizontal and vertical axes are the 
time points of real and predicted EEG.
"""

import scipy.io
import numpy as np

print('')
print('>>> Sleemory preprocessed data info <<<')

# =============================================================================
# Read Sleemory preprocessed data
# =============================================================================

data = scipy.io.loadmat('dataset/temp_sleemory/preprocessed_data/sleemory_localiser_dataset.mat')
print(data.keys())

prepr_data = data['ERP_all']
imgs_all = data['imgs_all']
ch_names = []
for ch in data['channel_names'][:,0]:
    ch_names.append(ch[0])
times = data['time']

print('Original data shape: ', prepr_data.shape)
print(len(ch_names))
print(ch_names)
print(times.shape, times[0,-1])
print(imgs_all.shape, len(np.unique(imgs_all)))

# =============================================================================
# Drop extra channels
# =============================================================================

index_Fz = ch_names.index('Fz')
index_Fpz = ch_names.index('Fpz')

prepr_data = np.delete(prepr_data, [index_Fz, index_Fpz], axis=1)
print('New data shape: ', prepr_data.shape)