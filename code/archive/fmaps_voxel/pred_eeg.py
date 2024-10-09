import pickle
import os
import scipy
import argparse
import numpy as np

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--networks', default=None, type=str)
parser.add_argument('--num_feat', default=1000, type=int)
parser.add_argument('--sub',      default=None,    type=int) 
args = parser.parse_args()

print('')
print(f'>>> Predict EEG from the encoding model ({args.networks}) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')

# =============================================================================
# Load the encoding model
# =============================================================================

model_dir = f'output/sleemory_localiser_vox/model/'
reg_model_fname = f'{args.networks}_reg_model.pkl'
print(reg_model_fname)
reg = pickle.load(open(f'{model_dir}/reg_model/sub-{args.sub}/{reg_model_fname}', 'rb'))

# =============================================================================
# Load the test fmaps
# =============================================================================

fmaps_data = scipy.io.loadmat(f'output/sleemory_retrieval_vox/dnn_feature_maps/best_feature_maps/sub_{args.sub}/{args.networks}-best-{args.num_feat}_fmaps.mat')

# Load labels
fmaps_labels = np.char.rstrip(fmaps_data['imgs_all'])
print(fmaps_labels)
print(fmaps_labels.shape)
# fmaps_labels = fmaps_labels.flatten().tolist()
# print(fmaps_labels)

# Load fmaps
fmaps = fmaps_data['fmaps']
print(fmaps.shape)

# =============================================================================
# Prediction 
# =============================================================================

pred_eeg = reg.predict(fmaps)
pred_eeg = pred_eeg.reshape(pred_eeg.shape[0], 3294, 301) # (img, voxel, time)
print(pred_eeg.shape)

# =============================================================================
# Save
# =============================================================================

save_dir = f'output/sleemory_retrieval_vox/pred_eeg/{args.networks}/'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
pred_eeg_fname = f'{args.networks}_sub-{args.sub:03d}_pred_eeg.mat'
print(f'{pred_eeg_fname} is saved in folder: {save_dir}')
scipy.io.savemat(f'{save_dir}/{pred_eeg_fname}', {'pred_eeg': pred_eeg, 
												  'imgs_all': fmaps_labels}) 