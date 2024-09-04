import os
import scipy.io
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
import argparse
import mat73

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--networks', default=None, type=str)
# If it's Alexnet, specify the layer name
parser.add_argument('--num_feat', default=None, type=int) 
parser.add_argument('--sub', default=None, type=int) 
args = parser.parse_args()

print('')
print(f'>>> Train the encoding model <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')
 
        


### Load the training DNN feature maps ###
fmaps_fname = f'{args.networks}-best-{args.num_feat}_fmaps.mat'
print(fmaps_fname)
fmaps_train = scipy.io.loadmat(f'output/sleemory_localiser_vox/dnn_feature_maps/{fmaps_fname}')
print(fmaps_train.shape)

# Load fmaps
fmaps = fmaps_train['fmaps'] # (img, 'num_token', num_feat)
if args.networks == 'BLIP-2': # Need to select token or mean pooling
    fmaps = np.mean(fmaps, axis=1)

### load labels ###
fmap_labels = fmaps_train['imgs_all']



### Load the training EEG data ###
eeg_dir = '/home/simon/Documents/gitrepos/shannon_encodingmodelsEEG/dataset/sleemory_localiser/preprocessed_data'

fname = f'sub-{args.sub:03d}_task-localiser_source_data'
data = mat73.loadmat(os.path.join(eeg_dir, fname+'.mat'))
eeg = data['sub_eeg_loc']['eeg']
eeg = np.reshape(eeg, (eeg.shape[0], -1)) # (img, ch x time)

eeg_labels = data['sub_eeg_loc']['images']
eeg_labels = [s[0].split('.')[0] for s in eeg_labels]
eeg_labels = np.asarray(eeg_labels)

del data

# Check the order of two labels
print(eeg_labels == fmap_labels)

reorder_fmaps = np.empty(fmaps.shape)
for eeg_label in eeg_labels:
	fmaps_idx = np.where(fmap_labels == eeg_label)[0]
	reorder_fmaps = fmaps[fmaps_idx]
	

### Train the encoding model ###
# Train the encoding models
reg = LinearRegression().fit(fmap, eeg)
# Save the model
model_dir = f'output/sleemory_localiser_vox/model/reg_model/sub-{args.sub}'
if os.path.isdir(model_dir) == False:
	os.makedirs(model_dir)
reg_model_fname = f'{args.networks}_reg_model.pkl'
print(reg_model_fname)
pickle.dump(reg, open(f'{model_dir}/{reg_model_fname}','wb'))