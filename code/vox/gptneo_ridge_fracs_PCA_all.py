import os
import scipy.io
import numpy as np
from fracridge import FracRidgeRegressorCV
# from sklearn.linear_model import LinearRegression
import argparse
import mat73
from tqdm import tqdm
from sklearn.linear_model import Ridge
from func import *
from joblib import Parallel, delayed

# =============================================================================
# Input arguments
# =============================================================================

networks = 'GPTNeo'

parser = argparse.ArgumentParser()
# parser.add_argument('--sub', default=None, type=int)
parser.add_argument('--whiten', default=False, type=bool)
# parser.add_argument('--num_compo', default=1000, type=int)
args = parser.parse_args()

print('')
print(f'>>> Train the encoding (Ridge) model ({networks}) with PCA across subjects <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')

# =============================================================================
# Load raww fmaps
# =============================================================================

# Load localiser fmaps
fmaps, fmap_labels = load_GPTNeo_fmaps('localiser')

# Load retrieval fmaps
retri_fmaps, retri_flabels = load_GPTNeo_fmaps('retrieval')

# =============================================================================
# Load EEG
# =============================================================================

all_subs_eeg = {}
all_subs_eeg_labels = {}

sub_start, sub_end = 2, 27
for sub in range(sub_start, sub_end):
	
	if sub == 17:
		pass
	else:
        # Load single EEG
		print(f'sub {sub}')
		eeg, eeg_labels = load_eeg_to_train_enc(sub, whiten=args.whiten)
	    
		all_subs_eeg[f'sub_{sub}'] = eeg
		all_subs_eeg_labels[f'sub_{sub}'] = eeg_labels

# =============================================================================
# Get all flabels 
# =============================================================================

for sub in range(sub_start, sub_end):
	if sub == 17:
		pass
	else:
		eeg = np.squeeze(all_subs_eeg[f'sub_{sub}'][:, :, 0])
		eeg_labels = all_subs_eeg_labels[f'sub_{sub}']

		# Reorder localiser fmaps
		reorder_fmaps, reorder_flabels = customize_fmaps(eeg, eeg_labels, fmaps, fmap_labels)
		
		# Concatenate eeg per time
		if sub == 2:
			tot_reorder_fmaps = reorder_fmaps
			tot_reorder_flabels = reorder_flabels
		else:
			tot_reorder_fmaps = np.concatenate((tot_reorder_fmaps, reorder_fmaps), axis=0)
			tot_reorder_flabels = np.concatenate((tot_reorder_flabels, reorder_flabels), axis=0)

print(tot_reorder_fmaps.shape, tot_reorder_flabels.shape)

# =============================================================================
# Extract the best features (PCA)
# =============================================================================

# Concatenate localiser and retrieval fmaps
tot_fmaps = np.concatenate([retri_fmaps, tot_reorder_fmaps], axis=0)
# tot_flabels = np.concatenate(retri_flabels, tot_reorder_flabels)
print(tot_fmaps.shape)

from sklearn.decomposition import PCA
pca = PCA(n_components=args.num_compo)
tot_fmaps = pca.fit(tot_fmaps).transform(tot_fmaps)
print(tot_fmaps.shape)

# =============================================================================
# Iterate over time 
# =============================================================================

# n_alphas = 20
fracs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])

num_time = 301
num_vox = 3294

# # Parallel
# def fit_frr(timepoint):

#     # Get all EEG at each time point
# 	for sub in range(2, 27):
# 		if sub == 17:
# 			pass
# 		else:
# 			eeg_t = np.squeeze(all_subs_eeg[f'sub_{sub}'][:, :, timepoint])

# 			# Concatenate eeg per time
# 			if sub == 2:
# 				tot_eeg_t = eeg_t # (trials, voxels,)
# 			else:
# 				tot_eeg_t = np.concatenate((tot_eeg_t, eeg_t), axis=0)

# 	frr = FracRidgeRegressorCV()
# 	frr.fit(tot_fmaps[4:], tot_eeg_t, frac_grid = fracs)
# 	pred_eeg = frr.predict(tot_fmaps[:4])
# 	return pred_eeg

# pred_results = Parallel(n_jobs=-1)(delayed(fit_frr)(t) for t in tqdm(range(num_time)))
# pred_eeg = np.stack(pred_results, axis=-1)
# print(pred_eeg.shape) 



pred_eeg = np.empty((4, num_vox, num_time))
for t in tqdm(range(num_time))[:1]:
	for ivox in range(num_vox):

		# Get all EEG at each time point at each voxel
		for sub in range(2, 27):
			if sub == 17:
				pass
			else:
				eeg_t = np.squeeze(all_subs_eeg[f'sub_{sub}'][:, ivox, t])

				# Concatenate eeg per time
				if sub == 2:
					tot_eeg_t = eeg_t # (trials, voxels,)
				else:
					tot_eeg_t = np.concatenate((tot_eeg_t, eeg_t), axis=0)
		print(tot_eeg_t.shape)

		frr = FracRidgeRegressorCV()
		frr.fit(tot_fmaps[4:], tot_eeg_t, frac_grid = fracs)
		pred_eeg_t = frr.predict(tot_fmaps[:4])
		pred_eeg[:, ivox, t] = pred_eeg_t
		del frr

# print(pred_eeg)



# # save pred eeg
# save_dir = f'output/sleemory_retrieval_vox/pred_eeg_ridge_PCAall_whiten{args.whiten}/{networks}/'
# if os.path.isdir(save_dir) == False:
# 	os.makedirs(save_dir)
# scipy.io.savemat(f'{save_dir}/{networks}_pred_eeg.mat', {'pred_eeg': pred_eeg,
# 										                'imgs_all': retri_flabels})