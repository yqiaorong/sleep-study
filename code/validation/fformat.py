import scipy.io
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--networks', default=None, type=str)
parser.add_argument('--whiten', default=False, type=bool)
args = parser.parse_args()

load_dir = f'output/sleemory_localiser_vox/validation_test/corr_ridge_PCA_whiten{args.whiten}/{args.networks}/'
fname_format = args.networks+'_corr_trial_sub-{:03d}'
sub_i, sub_f = 2, 27
fnames = [fname_format.format(sub) for sub in range(sub_i, sub_f) if sub != 17]

for ifname, fname in enumerate(tqdm(fnames)):
    # Change file format from .npy to .mat
    corr = np.load(f'{load_dir}/{fname}.npy')
    scipy.io.savemat(f'{load_dir}/{fname}.mat', {'corr': corr})
    
    # # Sanity check
    # corr = scipy.io.loadmat(f'{load_dir}/{fname}.mat')
    # print(corr.keys())