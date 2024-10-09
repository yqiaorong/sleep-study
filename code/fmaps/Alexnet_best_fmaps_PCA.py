import os
import argparse
import scipy
import numpy as np
from tqdm import tqdm

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--layer',      default='all', type=str)
args = parser.parse_args()

print('')
print('Extract sleemory images feature maps AlexNet <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))



layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']

# Load localiser
fmaps_dir = f'/home/simon/Documents/gitrepos/shannon_encodingmodelsEEG/sleep-study/dataset/sleemory_localiser/dnn_feature_maps/full_feature_maps/alexnet/pretrained-{str(args.pretrained)}/'
fmaps_list = os.listdir(fmaps_dir)
for ifname, fname in enumerate(tqdm(fmaps_list, desc='localiser')):
    data = scipy.io.loadmat(fmaps_dir+fname)

    # Concatenate within each image
    if args.layer == 'all':
        for ilayer, layer in enumerate(layers):
            data[layer] = data[layer].reshape(1, -1)
            if ilayer == 0:
                fmaps_img = data[layer]
            else:
                fmaps_img = np.concatenate([fmaps_img, data[layer]], axis=1)
    else:
        fmaps_img = data[args.layer].reshape(1, -1)

    # Concatenate images
    if ifname == 0:
        local_fmaps = fmaps_img
    else:
        local_fmaps = np.concatenate([local_fmaps, fmaps_img]) # (img, feats.)
local_flabels = [fname[:-6] for fname in fmaps_list]

# Load retrieval
fmaps_dir = f'/home/simon/Documents/gitrepos/shannon_encodingmodelsEEG/sleep-study/dataset/sleemory_retrieval/dnn_feature_maps/full_feature_maps/alexnet/pretrained-{str(args.pretrained)}/'
fmaps_list = os.listdir(fmaps_dir)
for ifname, fname in enumerate(tqdm(fmaps_list, desc='retrieval')):
    data = scipy.io.loadmat(fmaps_dir+fname)

    # Concatenate within each image
    if args.layer == 'all':
        for ilayer, layer in enumerate(layers):
            data[layer] = data[layer].reshape(1, -1)
            if ilayer == 0:
                fmaps_img = data[layer]
            else:
                fmaps_img = np.concatenate([fmaps_img, data[layer]], axis=1)
    else:
        fmaps_img = data[args.layer].reshape(1, -1)

    # Concatenate images
    if ifname == 0:
        retri_fmaps = fmaps_img
    else:
        retri_fmaps = np.concatenate([retri_fmaps, fmaps_img]) # (img, feats.)
retri_flabels = [fname[:-6] for fname in fmaps_list]



# Concatenate localiser and retrieval fmaps
fmaps = np.concatenate([retri_fmaps, local_fmaps])
print(fmaps.shape)


# Apply PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=250)
tot_fmaps = pca.fit(fmaps).transform(fmaps)
print(tot_fmaps.shape)
del fmaps

# Split fmaps back
retri_fmaps = tot_fmaps[:4]
local_fmaps = tot_fmaps[4:]
print(retri_fmaps.shape, local_fmaps.shape)

# Save fmaps 
save_dir = 'dataset/sleemory_localiser/dnn_feature_maps/PCA_feature_maps/'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
scipy.io.savemat(f'{save_dir}/alexnet-{args.layer}_PCA_fmaps.mat', {'fmaps': local_fmaps, 'imgs_all': local_flabels})
scipy.io.savemat(f'{save_dir}/alexnet-{args.layer}_PCA_fmaps.mat', {'fmaps': retri_fmaps, 'imgs_all': retri_flabels})