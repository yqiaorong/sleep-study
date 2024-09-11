"""This script train the encoding model per voxel."""
import os
import scipy.io
import numpy as np
from sklearn.linear_model import LinearRegression
import argparse
import mat73
from tqdm import tqdm
from sklearn.feature_selection import SelectKBest, f_regression
from func import *#

# =============================================================================
# Input arguments
# =============================================================================

networks = 'GPTNeo'

parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=None, type=int)
parser.add_argument('--whiten', default=False, type=bool)
parser.add_argument('--num_feat', default=1000, type=int)
args = parser.parse_args()

print('')
print(f'>>> Train the encoding model ({networks}) per time <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')

# =============================================================================
# Load eegs
# =============================================================================

# Load localiser fmaps
fmaps, fmap_labels = load_GPTNeo_fmaps('localiser')

# Load retrieval fmaps
retri_fmaps, retri_flabels = load_GPTNeo_fmaps('retrieval')

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
# for t_idx in tqdm(range(162, 163), desc='temporal encoding'):
	# print(f'vox {vox_idx}:')
	for sub in range(sub_start, sub_end):
		if sub == 17:
			pass
		else:
			eeg = np.squeeze(all_subs_eeg[f'sub_{sub}'][:, :, t_idx]) # (trials, voxels)
			eeg_labels = all_subs_eeg_labels[f'sub_{sub}']

           	# Reorder localiser fmaps
			reorder_fmaps, reorder_flabels = customize_fmaps(eeg, eeg_labels, fmaps, fmap_labels)
		    
			# Concatenate eeg per time
			tot_eeg_vox = eeg            # (trials, voxels,)
			tot_eeg_labels = eeg_labels  # (trials, feats,)

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
	# Train the encoding model per time
	# =============================================================================

	# print('Train the encoding model...')

	from sklearn.preprocessing import StandardScaler
	scalar = StandardScaler()
	best_localiser_fmaps = scalar.fit_transform(best_localiser_fmaps)
	best_retri_fmaps = scalar.transform(best_retri_fmaps)
	# print(np.isnan(np.any(best_localiser_fmaps)))
	# print(np.isinf(np.any(best_localiser_fmaps)))
	# from matplotlib import pyplot as plt
	# plt.figure()
	# plt.hist(best_localiser_fmaps)
	# plt.show()

	# Build the model
	reg = LinearRegression().fit(best_localiser_fmaps, tot_eeg_vox)

	# Pred eeg per voxel
	pred_eeg = reg.predict(best_retri_fmaps)
	# print(pred_eeg.shape)

	tot_pred_eeg.append(pred_eeg)
	del best_localiser_fmaps, best_retri_fmaps, tot_eeg_vox
	del feature_selection, scalar, reg
	
tot_pred_eeg = np.array(tot_pred_eeg).swapaxes(0, 1).swapaxes(1, 2)
print(tot_pred_eeg.shape)

# save pred eeg
save_dir = f'output/sleemory_retrieval_vox/pred_eeg_time_whiten{args.whiten}/{networks}/'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
# np.save(save_dir+f'ResNet_fc_pred_eeg', {'pred_eeg': tot_pred_eeg,
# 										 'imgs_all': retri_flabels})
scipy.io.savemat(f'{save_dir}/{networks}_pred_eeg_sub-{args.sub:03d}.mat', {'pred_eeg': tot_pred_eeg,
										                               'imgs_all': retri_flabels})