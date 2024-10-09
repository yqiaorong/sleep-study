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
parser.add_argument('--networks', default=None, type=str) # [gpteno / ]
parser.add_argument('--num_feat', default=1000, type=int) 
parser.add_argument('--sub',      default=None, type=int) 
args = parser.parse_args()

print('')
print(f'>>> Train the encoding model ({args.networks}) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')
 
# =============================================================================
# Load the localiser fmaps (checked)
# =============================================================================

fmaps_fname = f'{args.networks}-best-{args.num_feat}_fmaps.mat'
fmaps_path = f'output/sleemory_localiser_vox/dnn_feature_maps/best_feature_maps/sub_{args.sub}'
print(fmaps_path)
print(fmaps_fname)
fmaps_data = scipy.io.loadmat(f'{fmaps_path}/{fmaps_fname}')
print('fmaps successfully loaded')

# Load fmaps
fmaps = fmaps_data['fmaps'] # (img, 'num_token', num_feat)
print(fmaps.shape)

# load labels (contains .jpg)
fmap_labels = np.char.rstrip(fmaps_data['imgs_all'])
fmap_labels = np.squeeze(fmap_labels)
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

# # Check the order of two labels
# if np.all(eeg_labels == fmap_labels) == False:
# 	reorder_fmaps = np.empty(fmaps.shape)
# 	for eeg_label in eeg_labels:
# 		fmaps_idx = np.where(fmap_labels == eeg_label)[0]
# 		reorder_fmaps = fmaps[fmaps_idx]
# else:
#     reorder_fmaps = fmaps
# print(reorder_fmaps.shape, eeg.shape)

# =============================================================================
# Check the order of two labels (checked)
# =============================================================================

if np.all(eeg_labels == fmap_labels) == False:
	print("The labels order don't match")
	print('Cannot train the encoding model!!')
else:
	print("The labels order match")

	# =============================================================================
	# Train the encoding model
	# =============================================================================

	print('Train the encoding model...')
	reg = LinearRegression().fit(fmaps, eeg)
	# Save the model
	model_dir = f'output/sleemory_localiser_vox/model/reg_model/sub-{args.sub}/'
	if os.path.isdir(model_dir) == False:
		os.makedirs(model_dir)
	reg_model_fname = f'{args.networks}_reg_model.pkl'
	print(reg_model_fname)
	pickle.dump(reg, open(f'{model_dir}/{reg_model_fname}','wb'))
	print('The encoding model saved!')