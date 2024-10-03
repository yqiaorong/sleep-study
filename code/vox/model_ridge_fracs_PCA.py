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
parser.add_argument('--networks',  default=None, type=str)
parser.add_argument('--layer_name',default='',   type=str)
parser.add_argument('--sub',       default=None, type=int)
parser.add_argument('--whiten',    default=False,type=bool)
args = parser.parse_args()

if args.layer_name == '':
	model_name = args.networks
else:
	model_name = args.networks + '-' + args.layer_name

print('')
print(f'>>> Train the encoding (Ridge) model ({model_name}) with PCA per subject <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')

# =============================================================================
# Load raww fmaps
# =============================================================================

if args.networks == 'GPTNeo':
	fmaps, fmap_labels = load_GPTNeo_fmaps('localiser')
	retri_fmaps, retri_flabels = load_GPTNeo_fmaps('retrieval')
elif args.networks == 'mpnet':
   fmaps, fmap_labels = load_mpnet_fmaps('localiser')
   retri_fmaps, retri_flabels = load_mpnet_fmaps('retrieval')
elif args.networks == 'Alexnet':
	fmaps, fmap_labels = load_AlexNet_fmaps('localiser', layer=args.layer)
	fmap_labels = np.array([item +'.jpg' for item in fmap_labels])
	retri_fmaps, retri_flabels = load_AlexNet_fmaps('retrieval', layer=args.layer)
	retri_flabels = np.array([item +'.jpg' for item in retri_flabels])
elif args.networks == 'ResNet':
	fmaps, fmap_labels = load_ResNet_fmaps('localiser', args.layer_name)
	retri_fmaps, retri_flabels = load_ResNet_fmaps('retrieval', args.layer_name)

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

# Concatenate localiser and retrieval fmaps
tot_fmaps = np.concatenate([retri_fmaps, tot_reorder_fmaps], axis=0)
# tot_flabels = np.concatenate(retri_flabels, tot_reorder_flabels)
print(tot_fmaps.shape)

if args.networks == 'Alexnet':
	pass 
else:
	from sklearn.decomposition import PCA
	pca = PCA(n_components=250)
	tot_fmaps = pca.fit(tot_fmaps).transform(tot_fmaps)
print(tot_fmaps.shape)

# =============================================================================
# Iterate over time 
# =============================================================================

fracs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])

num_time = 301
num_vox = 3294

def fit_frr(timepoint):
	frr = FracRidgeRegressorCV()
	frr.fit(tot_fmaps[4:], tot_eeg[:,:,timepoint], frac_grid = fracs)
	pred_eeg = frr.predict(tot_fmaps[:4])
	return pred_eeg
	

pred_results = Parallel(n_jobs=-1)(delayed(fit_frr)(t) for t in tqdm(range(num_time), desc='pred EEG'))
pred_eeg = np.stack(pred_results, axis=-1)
print(pred_eeg.shape) 

# save pred eeg
save_dir = f'output/sleemory_retrieval_vox/pred_eeg_ridge_PCA_whiten{args.whiten}/{model_name}/'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
scipy.io.savemat(f'{save_dir}/{model_name}_pred_eeg_sub-{args.sub:03d}.mat', {'pred_eeg': pred_eeg,
										                                      'imgs_all': retri_flabels})