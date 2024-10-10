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

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--networks',  default=None, type=str)
parser.add_argument('--layer_name',default='',   type=str)
parser.add_argument('--n_permu',   default=1000, type=int)
parser.add_argument('--whiten',    default=False,type=bool)
args = parser.parse_args()

if args.layer_name == '':
	model_name = args.networks
else:
	model_name = args.networks + '-' + args.layer_name

print('')
print(f'>>> Permutation test of the encoding (Ridge) model ({model_name}) with PCA per subject <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')

# =============================================================================
# Load raww fmaps
# =============================================================================

# Load localiser fmaps
if args.networks == 'GPTNeo':
    fmaps, fmap_labels = load_GPTNeo_fmaps('localiser')
elif args.networks == 'Alexnet':
	fmaps, fmap_labels = load_AlexNet_fmaps('localiser', layer=args.layer_name)
else: 
   fmaps, fmap_labels = load_fmaps('localiser', model_name)



num_time = 301
num_vox = 3294
corr_tot = np.zeros((args.n_permu, num_vox, num_time))

# Save dir
save_dir = f'output/sleemory_localiser_vox/permutation_test/'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)



# Check the last saved sub
if os.path.exists(f'{save_dir}/{model_name}_corr_chunk-0.mat'):
	for i in range(5):
		data = scipy.io.loadmat(f'{save_dir}/{model_name}_corr_chunk-{i}.mat')
		if i == 0:
			end_sub = data['end_sub']
			if end_sub == 26:
				print('All finished!')
				# os._exit(0)
			start_sub = end_sub+1
		corr_tot[i*200:(i+1)*200] = data['corr']
		del data
	print(f'Continue from the last saved sub: {end_sub[0][0]}...')
	os._exit(0)
else:
	start_sub = 2
	print(f'New start from sub: {start_sub}...')



num_sub = 24
for sub in range(start_sub, 27):
	if sub != 17:
		
		# =============================================================================
		# Load EEG
		# =============================================================================

		tot_eeg, tot_eeg_labels, tot_reorder_fmaps, tot_reorder_flabels = load_and_match_eeg_and_fmaps(sub, args, fmaps, fmap_labels)

		# =============================================================================
		# Extract the best features (PCA)
		# =============================================================================

		if args.networks == 'Alexnet':
			pass
		else:
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

		def fit_frr(timepoint):
			frr = FracRidgeRegressorCV()
			frr.fit(tot_fmaps_train, tot_eeg_train[:,:,timepoint], frac_grid = fracs)
			pred_eeg = frr.predict(tot_fmaps_test)
			return pred_eeg

		pred_results = Parallel(n_jobs=-1)(delayed(fit_frr)(t) for t in tqdm(range(num_time), desc='pred EEG'))
		pred_eeg = np.stack(pred_results, axis=-1)
		print(pred_eeg.shape) 

		# =============================================================================
		# Correlations
		# =============================================================================
		
		def permu(ipermu):
			corr = np.zeros((1, num_vox, num_time))
			for itime in range(num_time):
				for ivox in range(num_vox):
					rand_eeg = np.random.permutation(tot_eeg_test[:, ivox, itime])
					corr[0, ivox, itime] = np.corrcoef(pred_eeg[:, ivox, itime], rand_eeg)[0, 1]
			return corr
		
		# def permu(ipermu):
		# 	corr = np.zeros((1, num_vox, num_time))
		# 	for itime in range(num_time):
		# 		for ivox in range(num_vox):
		# 			rand_eeg = np.random.permutation(tot_eeg_test[:, ivox, itime])
		# 			test_val = pd.Series(rand_eeg)
		# 			pred_val = pd.Series(pred_eeg[:, ivox, itime])
		# 			corr[0, ivox, itime] = test_val.corr(pred_val)
		# 	return corr
		
		permu_result = Parallel(n_jobs=-1)(delayed(permu)(ip) for ip in tqdm(range(args.n_permu), desc='permutations'))
		corr = np.squeeze(np.stack(permu_result, axis=0))
		del permu_result
		print(corr.shape)
		
		# Append to the total matrix
		corr_tot += corr
		del corr
		corr_tot_split = np.split(corr_tot, 5, axis=0)

		for idata, data in enumerate(corr_tot_split):
			print(data.shape)
			scipy.io.savemat(f'{save_dir}/{model_name}_corr_chunk-{idata}.mat', {'corr': data,
																		         'end_sub': sub})
			print(f'Until sub {sub}, chunk {idata} saved! ')
print('All saved! ')