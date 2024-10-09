import os
import scipy.io
import numpy as np
import pickle
import argparse

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=None, type=int) 
args = parser.parse_args()

networks = 'ResNet-fc'

print('')
print(f'>>> Predict EEG ({networks}) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')
 
# =============================================================================
# Load encoding model
# =============================================================================

reg_model_fname = f'{networks}_reg_model.pkl'
print(reg_model_fname)
model_dir = f'output/sleemory_localiser_vox/model/reg_model/sub-{args.sub}/'
reg = pickle.load(open(f'{model_dir}/{reg_model_fname}', 'rb'))

# =============================================================================
# Load the retrieval fmaps (checked)
# =============================================================================

fmaps_fname = f'{networks}_fmaps.mat'
print(fmaps_fname)
fmaps_path = f'dataset/sleemory_retrieval/dnn_feature_maps/full_feature_maps/{networks}/{fmaps_fname}'
print(fmaps_path)
fmaps_data = scipy.io.loadmat(fmaps_path)
print('fmaps successfully loaded')

# Load fmaps 
fmaps = fmaps_data['fmaps'] # (img, 'num_token', num_feat)
print(fmaps.shape)

# load labels (contains .jpg)
fmap_labels = np.char.rstrip(fmaps_data['imgs_all'])
print(fmap_labels.shape)
print(fmap_labels)

# =============================================================================
# Prediction 
# =============================================================================

pred_eeg = reg.predict(fmaps)
pred_eeg = pred_eeg.reshape(pred_eeg.shape[0], 3294, 301) # (img, voxel, time)
print(pred_eeg.shape)

# =============================================================================
# Save
# =============================================================================

save_dir = f'output/sleemory_retrieval_vox/pred_eeg/{networks}/'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
pred_eeg_fname = f'{networks}_sub-{args.sub:03d}_pred_eeg.mat'
print(f'{pred_eeg_fname} is saved in folder: {save_dir}')
scipy.io.savemat(f'{save_dir}/{pred_eeg_fname}', {'pred_eeg': pred_eeg, 
												  'imgs_all': fmap_labels}) 