import argparse
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import scipy.io

parser = argparse.ArgumentParser()
parser.add_argument('--networks', default=None, type=str)
parser.add_argument('--whiten', default=False, type=bool)
args = parser.parse_args()

load_dir = f'output/sleemory_localiser_vox/validation_test/corr_ridge_PCA_whiten{args.whiten}/{args.networks}/'
fname_format = args.networks+'_corr_trial_sub-{:03d}.mat'
sub_i, sub_f = 2, 27
fnames = [fname_format.format(sub) for sub in range(sub_i, sub_f) if sub != 17]

num_sub = sub_f - sub_i
num_vox, num_time = 3294, 301
tot_corr = np.zeros((num_sub, num_vox, num_time))
for ifname, fname in enumerate(tqdm(fnames)):
    data = scipy.io.loadmat(f'{load_dir}/{fname}')
    corr = data['corr']
    tot_corr[ifname] = corr

avg_corr = np.mean(tot_corr, axis=0)

# Plot
plt.figure()
plt.imshow(avg_corr, aspect='auto', cmap='viridis', origin='lower')
plt.colorbar(label='Corr Coeffs')
plt.xlabel('Time')
plt.ylabel('Voxel')
plt.savefig(f'{load_dir}/{args.networks}_corr_trial')
plt.close()

# Plot average across voxels
avg_corr_t = np.mean(avg_corr, axis=0)
t = np.linspace(-0.25, 1, 301)
plt.figure()
plt.plot(t, avg_corr_t, label=args.networks)
plt.xlabel('Time (s)')
plt.ylabel('Corr Coeffs')
plt.legend(loc='best')
plt.savefig(f'{load_dir}/{args.networks}_corr_trial_avg_across_vox')
plt.close()