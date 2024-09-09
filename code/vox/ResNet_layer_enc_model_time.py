"""This script train the encoding model per voxel."""
import os
import scipy.io
import numpy as np
from sklearn.linear_model import LinearRegression
import argparse
import mat73
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=None, type=int)
parser.add_argument('--whiten', default=True, type=bool)
parser.add_argument('--layer_name', default=None, type=str) # layer4.2.conv3 / fc
parser.add_argument('--num_feat', default=1000, type=int)
args = parser.parse_args()

networks = f'ResNet-{args.layer_name}'

print('')
print(f'>>> Train the encoding model ({networks}) per time <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')

# =============================================================================
# Func
# =============================================================================

def load_eeg_to_train_enc(sub, whiten = False):
	import pickle
	eeg_fname = f'sub-{sub:03d}_task-localiser_source_data'
	if whiten==False:
		eeg_dir = '/home/simon/Documents/gitrepos/shannon_encodingmodelsEEG/dataset/sleemory_localiser/preprocessed_data/'
		
		eeg_data = mat73.loadmat(os.path.join(eeg_dir, eeg_fname+'.mat'))
		eeg = eeg_data['sub_eeg_loc']['eeg'].astype(np.float32)
		print(f'Initial eeg shape {eeg.shape}')
		# eeg_vox = np.squeeze(eeg[:, vox_idx, :])
		# print(f'Initial eeg vox shape {eeg_vox.shape}')

		eeg_labels = eeg_data['sub_eeg_loc']['images']
		eeg_labels = [s[0] for s in eeg_labels]
		eeg_labels = np.asarray(eeg_labels)
		del eeg_data
	else:
		eeg_dir = 'output/sleemory_localiser_vox/whiten_eeg/'
		with open(os.path.join(eeg_dir, eeg_fname+'.pkl'), 'rb') as f:
			eeg_data = pickle.load(f)

			eeg = eeg_data['sub_eeg_loc']['eeg'].astype(np.float32)
			print(f'Initial eeg shape {eeg.shape}')
		
			# eeg = eeg_data['sub_eeg_loc/eeg']
			# eeg_labels = eeg_data['sub_eeg_loc/images']
			eeg_labels = eeg_data['sub_eeg_loc']['images']
			eeg_labels = [s[0] for s in eeg_labels]
			eeg_labels = np.asarray(eeg_labels)
			# print(eeg_labels)

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
def load_ResNet_fmaps(dataset, layer_name):
	fmaps_fname = f'ResNet-{layer_name}_fmaps.mat'
	print(fmaps_fname)
	fmaps_path = f'/home/simon/Documents/gitrepos/shannon_encodingmodelsEEG/sleep-study/dataset/sleemory_{dataset}/dnn_feature_maps/full_feature_maps/ResNet-{layer_name}/{fmaps_fname}'
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
# 
# =============================================================================
# Load eegs
# =============================================================================

# Load localiser fmaps
fmaps, fmap_labels = load_ResNet_fmaps('localiser', args.layer_name)

# Load retrieval fmaps
retri_fmaps, retri_flabels = load_ResNet_fmaps('retrieval', args.layer_name)

all_subs_eeg = {}
all_subs_eeg_labels = {}

# Load all eegs
sub_start, sub_end = args.sub, args.sub+1
for sub in range(sub_start, sub_end):
	
	if sub == 17:
		pass
	else:
        # Load eeg
		print(f'sub {sub}')
		eeg, eeg_labels = load_eeg_to_train_enc(sub, whiten=args.whiten)
	    
		all_subs_eeg[f'sub_{sub}'] = eeg
		all_subs_eeg_labels[f'sub_{sub}'] = eeg_labels

# =============================================================================
# Iterate over time
# =============================================================================

tot_pred_eeg = []
num_time = 301
for t_idx in tqdm(range(num_time), desc='temporal encoding'):
	# print(f'vox {vox_idx}:')
	for sub in range(sub_start, sub_end):
		if sub == 17:
			pass
		else:
			eeg = np.squeeze(all_subs_eeg[f'sub_{sub}'][:, :, t_idx]) # (trials, voxels)
			eeg_labels = all_subs_eeg_labels[f'sub_{sub}']

           	# Reorder localiser fmaps
			reorder_fmaps, reorder_flabels = customize_fmaps(eeg, eeg_labels, fmaps, fmap_labels)
		    
			# Concatenate eeg per voxel
			tot_eeg_vox = eeg
			tot_eeg_labels = eeg_labels

			tot_reorder_fmaps = reorder_fmaps
			tot_reorder_flabels = reorder_flabels

	# print(tot_eeg_vox.shape, tot_eeg_labels.shape)
	# print(tot_reorder_fmaps.shape, tot_reorder_flabels.shape)

	# =============================================================================
	# Extract the best features
	# =============================================================================
    
    # Build the feature selection model upon localiser fmaps
	tot_eeg_vox_fbest = np.mean(tot_eeg_vox, axis=1) # average across voxels
	feature_selection = SelectKBest(f_regression, k=args.num_feat).fit(tot_reorder_fmaps, tot_eeg_vox_fbest)
	del tot_eeg_vox_fbest

	# Select the best features of retrieval fmaps
	best_localiser_fmaps = feature_selection.transform(tot_reorder_fmaps)
	best_retri_fmaps = feature_selection.transform(retri_fmaps)
	# print(f'The final fmaps selected shape {best_localiser_fmaps.shape}, {best_retri_fmaps.shape}')
	del tot_reorder_fmaps

	# =============================================================================
	# Train the encoding model per voxel
	# =============================================================================

	# print('Train the encoding model...')
	
    # Standardization
	scalar = StandardScaler()
	best_localiser_fmaps = scalar.fit_transform(best_localiser_fmaps)
	best_retri_fmaps = scalar.transform(best_retri_fmaps)
    
	# Build the model
	reg = LinearRegression().fit(best_localiser_fmaps, tot_eeg_vox)

	# Pred eeg per voxel
	pred_eeg = reg.predict(best_retri_fmaps)
	# print(pred_eeg.shape)

	tot_pred_eeg.append(pred_eeg)
	del best_localiser_fmaps, best_retri_fmaps, tot_eeg_vox

tot_pred_eeg = np.array(tot_pred_eeg).swapaxes(0, 1).swapaxes(1, 2)
print(tot_pred_eeg.shape)

# save pred eeg
save_dir = f'output/sleemory_retrieval_vox/pred_eeg_time_whiten{args.whiten}/{networks}/'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
# np.save(save_dir+f'ResNet_fc_pred_eeg', {'pred_eeg': tot_pred_eeg,
# 										 'imgs_all': retri_flabels})
scipy.io.savemat(f'{save_dir}/{networks}_pred_eeg.mat', {'pred_eeg': tot_pred_eeg,
										                 'imgs_all': retri_flabels})