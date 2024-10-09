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
from func import *

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=None, type=int)
parser.add_argument('--whiten', default=False, type=bool)
parser.add_argument('--layer_name', default=None, type=str) # layer4.2.conv3 / fc
parser.add_argument('--num_feat', default=1000, type=int)
args = parser.parse_args()

networks = f'ResNet-{args.layer_name}'

print('')
print(f'>>> Train the encoding model ({networks}) per voxel <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')

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
# Iterate over voxels
# =============================================================================

tot_pred_eeg = []
num_vox = 3294
for vox_idx in tqdm(range(num_vox), desc='voxelwise encoding'):
	# print(f'vox {vox_idx}:')
	for sub in range(sub_start, sub_end):
		if sub == 17:
			pass
		else:
			eeg = np.squeeze(all_subs_eeg[f'sub_{sub}'][:, vox_idx, :])
			eeg_labels = all_subs_eeg_labels[f'sub_{sub}']

           	# Reorder localiser fmaps
			reorder_fmaps, reorder_flabels = customize_fmaps(eeg, eeg_labels, fmaps, fmap_labels)
		    
			# Concatenate eeg per voxel
			if sub == args.sub:
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
	# Extract the best features
	# =============================================================================
    
    # Build the feature selection model upon localiser fmaps
	tot_eeg_vox_fbest = np.mean(tot_eeg_vox[:,51:], axis=1) # Best feats only after baseline correction
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
	del feature_selection, scalar, reg

tot_pred_eeg = np.array(tot_pred_eeg).swapaxes(0, 1)
print(tot_pred_eeg.shape)

# save pred eeg
save_dir = f'output/sleemory_retrieval_vox/pred_eeg_voxelwise_whiten{args.whiten}/{networks}/'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
scipy.io.savemat(f'{save_dir}/{networks}_pred_eeg_sub-{args.sub:03d}.mat', {'pred_eeg': tot_pred_eeg,
										                                    'imgs_all': retri_flabels})