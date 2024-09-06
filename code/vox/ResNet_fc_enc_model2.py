import os
import scipy.io
import numpy as np
from sklearn.linear_model import LinearRegression
import argparse
import mat73
from tqdm import tqdm

# =============================================================================
# Input arguments
# =============================================================================

networks = 'ResNet-fc'

# parser = argparse.ArgumentParser()
# parser.add_argument('--vox_idx', default=0, type=int)
# args = parser.parse_args()

print('')
print(f'>>> Train the encoding model ({networks}) per voxel <<<')
print('\nInput arguments:')
# for key, val in vars(args).items():
# 	print('{:16} {}'.format(key, val))
# print('')

# =============================================================================
# Func
# =============================================================================

def load_eeg_to_train_enc(sub):
	eeg_dir = '/home/simon/Documents/gitrepos/shannon_encodingmodelsEEG/dataset/sleemory_localiser/preprocessed_data'

	eeg_fname = f'sub-{sub:03d}_task-localiser_source_data'
	eeg_data = mat73.loadmat(os.path.join(eeg_dir, eeg_fname+'.mat'))
	eeg = eeg_data['sub_eeg_loc']['eeg'].astype(np.float32)
	print(f'Initial eeg shape {eeg.shape}')
	# eeg_vox = np.squeeze(eeg[:, vox_idx, :])
	# print(f'Initial eeg vox shape {eeg_vox.shape}')

	eeg_labels = eeg_data['sub_eeg_loc']['images']
	eeg_labels = [s[0] for s in eeg_labels]
	eeg_labels = np.asarray(eeg_labels)
	del eeg_data

	return eeg, eeg_labels

def customize_fmaps(eeg, eeg_labels, fmaps_all, flabels_all):

	# Check the order of two labels
	reorder_fmaps = np.empty((eeg.shape[0], fmaps_all.shape[1]))
	reorder_flabels = []

	for idx, eeg_label in enumerate(eeg_labels):
		# print(eeg_label)
		fmaps_idx = np.where(flabels_all == eeg_label)[0]
		# print(f'fmaps idx: {fmaps_idx}')
		# print(flabels_all[fmaps_idx])
		if eeg_label == flabels_all[fmaps_idx]:
			# print('fmaps idx correct')
			reorder_fmaps[idx] = fmaps_all[fmaps_idx]
			reorder_flabels.append(flabels_all[fmaps_idx])
		else:
			print('fmaps idx incorrect')
		# print('')
	# print(reorder_fmaps.shape, eeg.shape)
	# print(f'Is there nan in reordered fmaps? {np.isnan(reorder_fmaps).any()}')

	# for idx in range(eeg_labels.shape[0]):
	# 	print(eeg_labels[idx], reorder_flabels[idx])
	return reorder_fmaps, np.squeeze(reorder_flabels)

# Load the feature maps
def load_ResNetfc_fmaps(dataset):
	fmaps_fname = f'ResNet-fc_fmaps.mat'
	print(fmaps_fname)
	fmaps_path = f'dataset/sleemory_{dataset}/dnn_feature_maps/full_feature_maps/ResNet-fc/{fmaps_fname}'
	print(fmaps_path)
	fmaps_data = scipy.io.loadmat(fmaps_path)
	print('fmaps successfully loaded')

	# Load fmaps 
	fmaps = fmaps_data['fmaps'].astype(np.float32) # (img, 'num_token', num_feat)
	print(fmaps.shape)

	# load labels (contains .jpg)
	fmap_labels = np.char.rstrip(fmaps_data['imgs_all'])
	print(fmap_labels.shape)

	return fmaps, fmap_labels

# =============================================================================
# Load eegs
# =============================================================================

# Load localiser fmaps
fmaps, fmap_labels = load_ResNetfc_fmaps('localiser')

# Load retrieval fmaps
retri_fmaps, retri_flabels = load_ResNetfc_fmaps('retrieval')

all_subs_eeg = {}
all_subs_eeg_labels = {}

# Load all eegs
for sub in range(2, 27):
	
	if sub == 17:
		pass
	else:
        # Load eeg
		print(f'sub {sub}')
		eeg, eeg_labels = load_eeg_to_train_enc(sub)
	    
		all_subs_eeg[f'sub_{sub}'] = eeg
		all_subs_eeg_labels[f'sub_{sub}'] = eeg_labels
print(all_subs_eeg[f'sub_2'])

# =============================================================================
# Iterate over voxels
# =============================================================================

tot_pred_eeg = []
num_vox = 3294
for vox_idx in tqdm(range(num_vox)):
	# print(f'vox {vox_idx}:')
	for sub in range(2, 27):
		if sub == 17:
			pass
		else:
			eeg = np.squeeze(all_subs_eeg[f'sub_{sub}'][:, vox_idx, :])
			eeg_labels = all_subs_eeg_labels[f'sub_{sub}']

           	# Reorder localiser fmaps
			reorder_fmaps, reorder_flabels = customize_fmaps(eeg, eeg_labels, fmaps, fmap_labels)
		    
			# Concatenate eeg per voxel
			if sub == 2:
				tot_eeg_vox = eeg
				tot_eeg_labels = eeg_labels

				tot_reorder_fmaps = reorder_fmaps
				tot_reorder_flabels = reorder_flabels
			else:
				tot_eeg_vox = np.concatenate((tot_eeg_vox, eeg), axis=0)
				tot_eeg_labels = np.concatenate((tot_eeg_labels, eeg_labels), axis=0)
      
				tot_reorder_fmaps = np.concatenate((tot_reorder_fmaps, reorder_fmaps), axis=0)
				tot_reorder_flabels = np.concatenate((tot_reorder_flabels, reorder_flabels), axis=0)

	# print(tot_eeg_vox.shape, tot_eeg_labels.shape)
	# print(tot_reorder_fmaps.shape, tot_reorder_flabels.shape)

	# =============================================================================
	# Train the encoding model per voxel
	# =============================================================================

	# print('Train the encoding model...')
	# Build the model
	reg = LinearRegression().fit(tot_reorder_fmaps, tot_eeg_vox)

	# Pred eeg per voxel
	pred_eeg = reg.predict(retri_fmaps)
	# print(pred_eeg.shape)

	tot_pred_eeg.append(pred_eeg)

tot_pred_eeg = np.array(tot_pred_eeg).swapaxes(0, 1)
# print(tot_pred_eeg.shape)

# save pred eeg
save_dir = 'output/sleemory_retrieval_vox/pred_eeg_voxelwise/'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
# np.save(save_dir+f'ResNet_fc_pred_eeg', {'pred_eeg': tot_pred_eeg,
# 										 'imgs_all': retri_flabels})
scipy.io.savemat(f'{save_dir}/ResNet_fc_pred_eeg.mat', {'pred_eeg': tot_pred_eeg,
										               'imgs_all': retri_flabels})