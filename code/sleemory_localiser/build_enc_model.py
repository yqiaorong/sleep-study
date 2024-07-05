import os
import scipy.io
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

model = 'CLIP'

if model == 'CLIP':
    ### Load the training DNN feature maps ###
    dnn_train_dir = os.path.join('dataset/sleemory_localiser/dnn_feature_maps')
    dnn_fmaps_train = scipy.io.loadmat(f'{dnn_train_dir}/CLIP_fmaps.mat')
    # Load fmaps
    fmap = dnn_fmaps_train['fmaps'] # (img, num_feat)
    # load labels
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
pickle.dump(reg, open(f'{model_dir}/CLIP_reg_model.pkl','wb'))