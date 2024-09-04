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
# If it's Alexnet, specify the layer name
parser.add_argument('--num_feat', default=None, type=int)
parser.add_argument('--sub', default=None, type=int) 
args = parser.parse_args()

print('')
print(f'>>> Predict EEG from the encoding model <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')

# =============================================================================
# Load best feat model
# =============================================================================
best_feat_model_name = f'{args.networks}_best_feat_model.pkl'
print(best_feat_model_name)
model_dir = f'output/sleemory_localiser_vox/model/'
freg = pickle.load(open(f'{model_dir}/best_feat_model/sub-{args.sub}/{best_feat_model_name}', 'rb'))

# =============================================================================
# Load encoding model
# =============================================================================

reg_model_fname = f'{args.networks}_reg_model.pkl'
print(reg_model_fname)
reg = pickle.load(open(f'{model_dir}/reg_model/sub-{args.sub}/{reg_model_fname}', 'rb'))

# =============================================================================
# Load full test fmaps
# =============================================================================

if args.networks == 'GPTNEO':
    fmaps_data = scipy.io.loadmat(f'dataset/sleemory_retrieval/dnn_feature_maps/full_feature_maps/GPTNeo/gptneo_fmaps.mat')

# Load labels
fmaps_labels = np.char.rstrip(fmaps_data['imgs_all'])

# Concatenate all layers
for ilayer, layer in enumerate(list(fmaps_data.keys())[3:]):
    if layer.startswith('layer'):
        if ilayer == 0:
            fmaps = fmaps_data[layer]
        else:
            fmaps = np.concatenate((fmaps, fmaps_data[layer]), axis=1)
print(f'fmaps all shape: {fmaps.shape}')

# =============================================================================
# Prediction 
# =============================================================================

# Select the best image features
best_fmaps = freg.transform(fmaps)
print(best_fmaps.shape)
pred_eeg = reg.predict(best_fmaps)
pred_eeg = pred_eeg.reshape(pred_eeg.shape[0], 3294, 301) # (img, voxel, time)
print(pred_eeg.shape)

# =============================================================================
# Save
# =============================================================================

save_dir = f'output/sleemory_retrieval_vox/pred_eeg/{args.networks}/'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
pred_eeg_fname = f'{args.networks}_sub-{args.sub}_pred_eeg.mat'
print(pred_eeg_fname)
scipy.io.savemat(f'{save_dir}/{pred_eeg_fname}', {'pred_eeg': pred_eeg}) 