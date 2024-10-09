import os
import scipy.io
import numpy as np
from scipy import stats as stats
from fracridge import FracRidgeRegressorCV
# from sklearn.linear_model import LinearRegression
import argparse
import mat73
from tqdm import tqdm
from sklearn.linear_model import Ridge
from func import *
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# =============================================================================
# Input arguments
# =============================================================================

networks = 'mpnet'

parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=None, type=int)
parser.add_argument('--whiten', default=False, type=bool)
args = parser.parse_args()

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
fmaps, fmap_labels = load_mpnet_fmaps('localiser')

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

eeg        = all_subs_eeg[f'sub_{args.sub}'] # (trials, voxels)
eeg_labels = all_subs_eeg_labels[f'sub_{args.sub}']

# Reorder localiser fmaps
reorder_fmaps, reorder_flabels = customize_fmaps(eeg, eeg_labels, fmaps, fmap_labels)


tot_eeg             = eeg              # (trials, voxels, time)
tot_eeg_labels      = eeg_labels       # (trials,)
tot_reorder_fmaps   = reorder_fmaps    # (trials, feats,)
tot_reorder_flabels = reorder_flabels  # (trials,)
del eeg, eeg_labels, reorder_fmaps, reorder_flabels

print(tot_eeg.shape, tot_eeg_labels.shape)
print(tot_reorder_fmaps.shape, tot_reorder_flabels.shape)

# =============================================================================
# Extract the best features (PCA)
# =============================================================================

from sklearn.decomposition import PCA
pca = PCA(n_components=250)
tot_reorder_fmaps = pca.fit(tot_reorder_fmaps).transform(tot_reorder_fmaps)
print(tot_reorder_fmaps.shape)

# =============================================================================
# Split the dataset
# =============================================================================

seed = 42
tot_eeg_train, tot_eeg_test, tot_fmaps_train, tot_fmaps_test, tot_flabels_train, tot_flabels_test = train_test_split(tot_eeg, tot_reorder_fmaps, tot_reorder_flabels, test_size=0.25, random_state=seed)
print(tot_eeg_train.shape, tot_eeg_test.shape, tot_fmaps_train.shape, tot_fmaps_test.shape, tot_flabels_train.shape, tot_flabels_test.shape)
del tot_eeg, tot_reorder_fmaps, tot_reorder_flabels

# =============================================================================
# Iterate over time 
# =============================================================================

# n_alphas = 20
fracs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])

num_time = 301
num_vox = 3294

def fit_frr(timepoint):
	frr = FracRidgeRegressorCV()
	frr.fit(tot_fmaps_train, tot_eeg_train[:,:,timepoint], frac_grid = fracs)
	pred_eeg = frr.predict(tot_fmaps_test)
	return pred_eeg

pred_results = Parallel(n_jobs=-1)(delayed(fit_frr)(t) for t in tqdm(range(num_time)))
pred_eeg = np.stack(pred_results, axis=-1)
print(pred_eeg.shape) 

# =============================================================================
# Correlations
# =============================================================================

num_test_trial = pred_eeg.shape[0]

corr_trial = np.zeros((num_vox, num_time))

# Correlation of image pattern
for itime in tqdm(range(num_time), desc='Correlations'):
	for ivox in range(num_vox):
		corr_trial[ivox, itime] = np.corrcoef(pred_eeg[:, ivox, itime], tot_eeg_test[:, ivox, itime])[0, 1]

# Save corr
save_dir = f'output/sleemory_localiser_vox/validation_test/corr_ridge_PCA_whiten{args.whiten}/{networks}/'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
scipy.io.savemat(f'{save_dir}/{networks}_corr_trial_sub-{args.sub:03d}.mat', {'corr': corr_trial})

# Plot
plt.figure()
plt.imshow(corr_trial, aspect='auto', cmap='winter', origin='lower')
plt.colorbar(label='Corr Coeffs')
plt.xlabel('Time')
plt.ylabel('Voxel')
plt.savefig(f'{save_dir}/{networks}_corr_trial_sub-{args.sub:03d}')
plt.close()