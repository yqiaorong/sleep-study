"""This script train the encoding model per voxel."""
import os
import scipy.io
import numpy as np
from sklearn.linear_model import LinearRegression
import argparse
import mat73
from tqdm import tqdm
from sklearn.feature_selection import SelectKBest, f_regression
from func import *

# =============================================================================
# Input arguments
# =============================================================================

networks = 'GPTNeo'

parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=None, type=int)
parser.add_argument('--whiten', default=False, type=bool)
# parser.add_argument('--num_feat', default=1000, type=int)
args = parser.parse_args()

print('')
print(f'>>> Train the encoding model ({networks}) with PCA <<<')
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



eeg = all_subs_eeg[f'sub_{args.sub}'] # (trials, voxels)
eeg_labels = all_subs_eeg_labels[f'sub_{args.sub}']

# Reorder localiser fmaps
reorder_fmaps, reorder_flabels = customize_fmaps(eeg, eeg_labels, fmaps, fmap_labels)


tot_eeg = eeg            # (trials, voxels,)
tot_eeg_labels = eeg_labels  # (trials, feats,)
tot_reorder_fmaps = reorder_fmaps
tot_reorder_flabels = reorder_flabels


# print(tot_eeg_vox.shape, tot_eeg_labels.shape)
# print(tot_reorder_fmaps.shape, tot_reorder_flabels.shape)

# =============================================================================
# Extract the best features (PCA)
# =============================================================================

# Concatenate localiser and retrieval fmaps
tot_fmaps = np.concatenate([retri_fmaps, tot_reorder_fmaps], axis=0)
# tot_flabels = np.concatenate(retri_flabels, tot_reorder_flabels)
print(tot_fmaps.shape)

from sklearn.decomposition import PCA
pca = PCA(n_components=250)
tot_fmaps = pca.fit(tot_fmaps).transform(tot_fmaps)
print(tot_fmaps.shape)

# =============================================================================
# Iterate over time 
# =============================================================================


num_time = 301
num_vox = 3294
pred_eeg = np.empty((4, num_vox, num_time))
for t_idx in tqdm(range(num_time), desc='temporal encoding'):

	# =============================================================================
	# Train the encoding model per time
	# =============================================================================

	# Build the model
	reg = LinearRegression().fit(tot_fmaps[4:], tot_eeg[:, :, t_idx])

	# Pred eeg per voxel per time
	pred_eeg[:, :, t_idx] = reg.predict(tot_fmaps[:4])
	del reg
print(pred_eeg)

# save pred eeg
save_dir = f'output/sleemory_retrieval_vox/pred_eeg_PCA_whiten{args.whiten}/{networks}/'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
# np.save(save_dir+f'ResNet_fc_pred_eeg', {'pred_eeg': tot_pred_eeg,
# 										 'imgs_all': retri_flabels})
scipy.io.savemat(f'{save_dir}/{networks}_pred_eeg_sub-{args.sub:03d}.mat', {'pred_eeg': pred_eeg,
										                               'imgs_all': retri_flabels})