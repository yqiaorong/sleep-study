save_dir = f'output/sleemory_localiser_vox/permutation_test/'

# import scipy
# data = scipy.io.loadmat('output/sleemory_localiser_vox/permutation_test/ResNet-fc_corr_all.mat')
# print(data.keys())

import h5py
with h5py.File(f'{save_dir}/ResNet-fc_corr_all.mat', 'r') as f:
    print(f.keys())