"""Preprocess the raw EEG data: re-reference, bandpass filter, epoching, 
baseline correction, frequency downsampling, multivariate noise normalization 
(MVNN),sorting of the data image conditions and reshaping the data to: 
Image conditions × EEG repetitions × EEG channels × EEG time points.
Then, the data of both test and training EEG partitions is saved.

Parameters
----------
subj : int
	Used subject.
n_ses : int
	Number of EEG sessions.
sfreq : int
	Downsampling frequency.
mvnn_dim : str
	Whether to compute the MVNN covariace matrices for each time point
	('time') or for each epoch/repetition ('epochs').
"""

import argparse
from func import epoching_THINGS2, mvnn_THINGS2, save_prepr_THINGS2

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subj', default=1, type=int)
parser.add_argument('--n_ses', default=4, type=int)
parser.add_argument('--sfreq', default=100, type=int)
parser.add_argument('--mvnn_dim', default='time', type=str)
args = parser.parse_args()

print('')
print('>>> THINGS EEG2 data preprocessing <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220

# =============================================================================
# Epoch and sort the data
# =============================================================================
epoched_test, _, ch_names, times = epoching_THINGS2(args, 'test', seed)
epoched_train, img_conditions_train, _, _ = epoching_THINGS2(args, 'training', seed)

# =============================================================================
# Multivariate Noise Normalization
# =============================================================================
whitened_test, whitened_train = mvnn_THINGS2(args, epoched_test, epoched_train)
del epoched_test, epoched_train

# =============================================================================
# Merge and save the preprocessed data
# =============================================================================
save_prepr_THINGS2(args, whitened_test, whitened_train, img_conditions_train, 
                   ch_names, times, seed)