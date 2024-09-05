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
parser.add_argument('--num_feat', default=None, type=int)
parser.add_argument('--sub', default=None, type=int) 
args = parser.parse_args()

print('')
print(f'>>> Predict EEG from the encoding model <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')



model_dir = f'output/sleemory_localiser_vox/model/'
if args.networks == 'GPTNEO' or args.networks == 'ResNet-fc':
    # =============================================================================
    # Load best feat model
    # =============================================================================
    best_feat_model_name = f'{args.networks}_best_feat_model.pkl'
    print(best_feat_model_name)
    
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

elif args.networks == 'ResNet':
    fmaps_data = scipy.io.loadmat(f'output/sleemory_retrieval_vox/dnn_feature_maps/best_feature_maps/sub_{args.sub}/ResNet-best-1000_fmaps.mat')
    
    # Load labels
    fmaps_labels = np.char.rstrip(fmaps_data['imgs_all'])

    # Load fmaps
    fmaps = fmaps_data['fmaps']
    
elif args.networks == 'ResNet-fc':
    fmaps_data = scipy.io.loadmat(f'output/sleemory_retrieval_vox/dnn_feature_maps/full_feature_maps/ResNet-fc/ResNet-fc_fmaps.mat')
    fmaps_labels = np.char.rstrip(fmaps_data['imgs_all'])
    fmaps = fmaps_data['fmaps']

print(f'fmaps all shape: {fmaps.shape}')
fmaps_labels = fmaps_labels.flatten().tolist()
print(fmaps_labels)

# =============================================================================
# Prediction 
# =============================================================================

if args.networks == 'GPTNEO' or args.networks == 'ResNet-fc':
    # Select the best image features
    fmaps = freg.transform(fmaps)
    print(fmaps.shape)
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
print(pred_eeg_fname)
scipy.io.savemat(f'{save_dir}/{pred_eeg_fname}', {'pred_eeg': pred_eeg, 'imgs_all': fmaps_labels}) 