import os
import scipy.io
import numpy as np
# from sklearn.linear_model import LinearRegression
import argparse
import mat73
from tqdm import tqdm
from sklearn.linear_model import Ridge
from func import *

# =============================================================================
# Input arguments
# =============================================================================

networks = 'GPTNeo'

parser = argparse.ArgumentParser()
# parser.add_argument('--sub', default=None, type=int)
parser.add_argument('--whiten', default=False, type=bool)
parser.add_argument('--num_compo', default=1000, type=int)
args = parser.parse_args()

print('')
print(f'>>> Train the encoding model ({networks}) with PCA <<<')
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

num_time = 301
num_vox = 3294
pred_eeg = np.empty((4, num_vox, num_time))
for t_idx in tqdm(range(num_time), desc='temporal encoding'):
    
	# Get all EEG at each time point
	for sub in range(sub_start, sub_end):
		if sub == 17:
			pass
		else:
			eeg = np.squeeze(all_subs_eeg[f'sub_{sub}'][:, :, t_idx])

			# Concatenate eeg per time
			if sub == 2:
				tot_eeg_vox = eeg            # (trials, voxels,)
				tot_eeg_labels = eeg_labels  # (trials, feats,)
			else:
				tot_eeg_vox = np.concatenate((tot_eeg_vox, eeg), axis=0)
				tot_eeg_labels = np.concatenate((tot_eeg_labels, eeg_labels), axis=0)
      
	# =============================================================================
	# Train the encoding model per time
	# =============================================================================

	# Build the model
	# reg = LinearRegression().fit(tot_fmaps[4:], tot_eeg_vox)
	reg = Ridge(alpha=1.0).fit(tot_fmaps[4:], tot_eeg_vox)
	# Pred eeg per voxel per time
	pred_eeg[:, :, t_idx] = reg.predict(tot_fmaps[:4])
	del reg
print(pred_eeg)

# save pred eeg
save_dir = f'output/sleemory_retrieval_vox/pred_eeg_ridge_PCAall-{args.num_compo}_whiten{args.whiten}/{networks}/'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
# np.save(save_dir+f'ResNet_fc_pred_eeg', {'pred_eeg': tot_pred_eeg,
# 										 'imgs_all': retri_flabels})
scipy.io.savemat(f'{save_dir}/{networks}_pred_eeg.mat', {'pred_eeg': pred_eeg,
										                'imgs_all': retri_flabels})