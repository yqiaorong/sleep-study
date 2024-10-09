save_dir = f'output/sleemory_localiser_vox/permutation_test/'
import scipy

for i in range(5):
    data = scipy.io.loadmat(f'{save_dir}/ResNet-layer3_corr_chunk-{i}.mat')
    print(data['end_sub'])