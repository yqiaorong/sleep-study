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
# If it's Alexnet, specify the layer name
parser.add_argument('--num_feat', default=None, type=int) 
# num_feat = -1 means using all features, If it's Alexnet, num_feat cannot be -1
args = parser.parse_args()

print('')
print(f'>>> Train the encoding model <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')

### Setup ###
if args.num_feat == -1:
    best_feat_cond = ''
    best_feat_whiten = ''
    reg_whiten = input('For reg model use whitened EEG or not? [T/F]')
    if reg_whiten == 'F':
        reg_whiten = ''
    elif reg_whiten == 'T':
        reg_whiten = 'whiten_'
else:
    best_feat_cond = f'-best-{args.num_feat}'
    best_feat_whiten = input('For best feat model use whitened EEG or not? [T/F]')
    if best_feat_whiten == 'F':
        best_feat_whiten = ''
    elif best_feat_whiten == 'T':
        best_feat_whiten = 'whiten_'
    reg_whiten = best_feat_whiten  
        


### Load the training DNN feature maps ###
dnn_fmaps_fname = f'{args.networks}{best_feat_cond}_{best_feat_whiten}fmaps.mat'
print(dnn_fmaps_fname)
dnn_fmaps_train = scipy.io.loadmat(f'dataset/sleemory_localiser/dnn_feature_maps/{dnn_fmaps_fname}')

# Load fmaps
fmap = dnn_fmaps_train['fmaps'] # (img, 'num_token', num_feat)
if args.networks == 'BLIP-2': # Need to select token or mean pooling
    fmap = np.mean(fmap, axis=1)

### load labels ###
fmap_labels = os.listdir('dataset/sleemory_localiser/image_set') # len(img)



### Load the training EEG data ###
eeg_train_fname = f'unique_{reg_whiten}eeg.mat'
print(eeg_train_fname)
eeg_data_train = scipy.io.loadmat(f'output/sleemory_localiser/whiten_eeg/{eeg_train_fname}')
eeg = eeg_data_train[f'{reg_whiten}eeg'] # (img, ch, time)
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
reg_model_fname = f'{args.networks}{best_feat_cond}_{reg_whiten}reg_model.pkl'
print(reg_model_fname)
pickle.dump(reg, open(f'{model_dir}/{reg_model_fname}','wb'))