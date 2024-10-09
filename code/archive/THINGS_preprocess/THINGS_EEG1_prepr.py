"""Preprocess the raw EEG data: re-reference, bandpass filter, epoching, 
baseline correction, frequency downsampling, multivariate noise normalization 
(MVNN),sorting of the data image conditions and reshaping the data to: 
Image conditions × EEG repetitions × EEG channels × EEG time points.
Then, the data of both test and training EEG partitions is saved.

Parameters
----------
subj : int
	Used subject.
sfreq : int
	Downsampling frequency.
mvnn_dim : str
	Whether to compute the MVNN covariace matrices for each time point
	('time') or for each epoch/repetition ('epochs').
"""

import argparse
from func import epoching_THINGS1, mvnn_THINGS1, save_prepr_THINGS1

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subj', default=1, type=int)
parser.add_argument('--sfreq', default=100, type=int)
parser.add_argument('--mvnn_dim', default='time', type=str)
args = parser.parse_args()

print('')
print('>>> THINGS EEG1 data preprocessing <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
 
# Set random seed for reproducible results
seed = 20200220

# =============================================================================
# Epoch and sort the data
# =============================================================================

epoched_test, ch_names, times = epoching_THINGS1(args)

# =============================================================================
# Multivariate Noise Normalization
# =============================================================================

whitened_test = mvnn_THINGS1(args, epoched_test)
del epoched_test

# =============================================================================
# Merge and save the preprocessed data
# =============================================================================

save_prepr_THINGS1(args, whitened_test, ch_names, times, seed)