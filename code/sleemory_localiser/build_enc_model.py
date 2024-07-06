import os
import scipy.io
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import argparse

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--networks', default=None, type=str)
args = parser.parse_args()

print('')
print(f'>>> Train the encoding model <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
 
 

### Load the training DNN feature maps ###
dnn_fmaps_train = scipy.io.loadmat(f'dataset/sleemory_localiser/dnn_feature_maps/{args.networks}_fmaps.mat')
# Load fmaps
fmap = dnn_fmaps_train['fmaps'] # (img, 'num_token', num_feat)
if args.networks == 'BLIP-2': # Need to select token or mean pooling
    fmap = np.mean(fmap, axis=1)

### load labels ###
fmap_labels = os.listdir('dataset/sleemory_localiser/image_set') # len(img)



### Load the training EEG data ###
eeg_train_dir = 'output/sleemory_localiser/whiten_eeg'
eeg_data_train = scipy.io.loadmat(f'{eeg_train_dir}/unique_whiten_eeg.mat')
eeg = eeg_data_train['whiten_eeg'] # (img, ch, time)
eeg = np.reshape(eeg, (eeg.shape[0], -1)) # (img, ch x time)
eeg_labels = eeg_data_train['unique_img'] # (img,)
del eeg_data_train



### Train the encoding model ###
# Train the encoding models
reg = LinearRegression().fit(fmap, eeg)
# Save the model
model_dir = 'dataset/sleemory_localiser/model/reg_model'
if os.path.isdir(model_dir) == False:
	os.makedirs(model_dir)
pickle.dump(reg, open(f'{model_dir}/{args.networks}_reg_model.pkl','wb'))