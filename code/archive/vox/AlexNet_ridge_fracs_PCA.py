import os
import scipy.io
from scipy import stats as stats
import numpy as np
from fracridge import FracRidgeRegressorCV
# from sklearn.linear_model import LinearRegression
import argparse
import mat73
from tqdm import tqdm
from sklearn.linear_model import Ridge
from func import *
from joblib import Parallel, delayed, cpu_count

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=None, type=int)
parser.add_argument('--whiten', default=False, type=bool)
parser.add_argument('--layer', default='all', type=str)
args = parser.parse_args()

networks = 'Alexnet'

print('')
print(f'>>> Train the encoding (Ridge) model ({networks}) with PCA per subject <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')

# =============================================================================
# Load raww fmaps
# =============================================================================

# Load localiser fmaps
fmaps, fmap_labels = load_AlexNet_fmaps('localiser', layer=args.layer)
fmap_labels = np.array([item +'.jpg' for item in fmap_labels])

# Load retrieval fmaps
retri_fmaps, retri_flabels = load_AlexNet_fmaps('retrieval', layer=args.layer)
retri_flabels = np.array([item +'.jpg' for item in retri_flabels])

# =============================================================================
# Load EEG
# =============================================================================

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


tot_eeg = eeg                # (trials, voxels,)
tot_eeg_labels = eeg_labels  # (trials, feats,)
tot_reorder_fmaps = reorder_fmaps
tot_reorder_flabels = reorder_flabels


# print(tot_eeg_vox.shape, tot_eeg_labels.shape)
# print(tot_reorder_fmaps.shape, tot_reorder_flabels.shape)

# Concatenate localiser and retrieval fmaps
tot_fmaps = np.concatenate([retri_fmaps, tot_reorder_fmaps], axis=0)
print(tot_fmaps.shape)

# =============================================================================
# Iterate over time 
# =============================================================================

# n_alphas = 20
# fracs = np.linspace(1/n_alphas, 1 + 1/n_alphas, n_alphas)
fracs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])

num_time = 301
num_vox = 3294

def fit_frr(timepoint):
	frr = FracRidgeRegressorCV()
	frr.fit(tot_fmaps[4:], tot_eeg[:,:,timepoint], frac_grid = fracs)
	pred_eeg = frr.predict(tot_fmaps[:4])
	return pred_eeg
	

pred_results = Parallel(n_jobs=-1)(delayed(fit_frr)(t) for t in tqdm(range(num_time)))
pred_eeg = np.stack(pred_results, axis=-1)
print(pred_eeg.shape) 

# save pred eeg
save_dir = f'output/sleemory_retrieval_vox/pred_eeg_ridge_PCA_whiten{args.whiten}/{networks}-{args.layer}/'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
# np.save(save_dir+f'ResNet_fc_pred_eeg', {'pred_eeg': tot_pred_eeg,
# 										 'imgs_all': retri_flabels})
scipy.io.savemat(f'{save_dir}/{networks}-{args.layer}_pred_eeg_sub-{args.sub:03d}.mat', {'pred_eeg': pred_eeg,
										                               'imgs_all': retri_flabels})