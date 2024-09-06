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
parser.add_argument('--sub', default=None, type=int) 
args = parser.parse_args()

networks = 'ResNet-fc'

print('')
print(f'>>> Train the encoding model ({networks}) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')
 
# =============================================================================
# Load the localiser fmaps (checked)
# =============================================================================

fmaps_fname = f'{networks}_fmaps.mat'
print(fmaps_fname)
fmaps_path = f'dataset/sleemory_localiser/dnn_feature_maps/full_feature_maps/{networks}/{fmaps_fname}'
print(fmaps_path)
fmaps_data = scipy.io.loadmat(fmaps_path)
print('fmaps successfully loaded')

# Load fmaps 
fmaps = fmaps_data['fmaps'] # (img, 'num_token', num_feat)
print(fmaps.shape)

# load labels (contains .jpg)
fmap_labels = np.char.rstrip(fmaps_data['imgs_all'])
print(fmap_labels.shape)

# =============================================================================
# Load the EEG data (checked)
# =============================================================================

eeg_dir = '/home/simon/Documents/gitrepos/shannon_encodingmodelsEEG/dataset/sleemory_localiser/preprocessed_data'

eeg_fname = f'sub-{args.sub:03d}_task-localiser_source_data'
eeg_data = mat73.loadmat(os.path.join(eeg_dir, eeg_fname+'.mat'))
eeg = eeg_data['sub_eeg_loc']['eeg']
print(f'Initial eeg shape {eeg.shape}')

eeg = np.reshape(eeg, (eeg.shape[0], -1)) # (img, ch x time)
print(f'To be trained eeg shape {eeg.shape}')

eeg_labels = eeg_data['sub_eeg_loc']['images']
eeg_labels = [s[0] for s in eeg_labels]
eeg_labels = np.asarray(eeg_labels)
del eeg_data

# =============================================================================
# Reorder fmaps (checked)
# =============================================================================

# Check the order of two labels
reorder_fmaps = np.empty((eeg.shape[0], fmaps.shape[1]))
reorder_flabels = []
for idx, eeg_label in enumerate(eeg_labels):
	print(eeg_label)
	fmaps_idx = np.where(fmap_labels == eeg_label)[0]
	print(f'fmaps idx: {fmaps_idx}')
	print(fmap_labels[fmaps_idx])
	if eeg_label == fmap_labels[fmaps_idx]:
		print('fmaps idx correct')
		reorder_fmaps[idx] = fmaps[fmaps_idx]
		reorder_flabels.append(fmap_labels[fmaps_idx])
	else:
		print('fmaps idx incorrect')
	print('')
print(reorder_fmaps.shape, eeg.shape)
print(reorder_fmaps)
print(f'Is there nan in reordered fmaps? {np.isnan(reorder_fmaps).any()}')

for idx in range(eeg_labels.shape[0]):
	print(eeg_labels[idx], reorder_flabels[idx])
del fmaps, fmap_labels

# =============================================================================
# Train the encoding model
# =============================================================================

print('Train the encoding model...')
reg = LinearRegression().fit(reorder_fmaps, eeg)
# Save the model
model_dir = f'output/sleemory_localiser_vox/model/reg_model/sub-{args.sub}/'
if os.path.isdir(model_dir) == False:
	os.makedirs(model_dir)
reg_model_fname = f'{networks}_reg_model.pkl'
print(reg_model_fname)
pickle.dump(reg, open(f'{model_dir}/{reg_model_fname}','wb'))
print('The encoding model saved!')