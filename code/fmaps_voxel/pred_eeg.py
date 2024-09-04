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
    
# Load encoding model
reg_model_fname = f'{args.networks}_reg_model.pkl'
print(reg_model_fname)
model_dir = f'output/sleemory_localiser_vox/model/reg_model/sub-{args.sub}'
reg = pickle.load(open(f'{model_dir}/{reg_model_fname}', 'rb'))


### Load test fmaps ###
fmaps_fname = f'{args.networks}-best-{args.num_feat}_fmaps.mat'
print(fmaps_fname)
fmaps_data = scipy.io.loadmat(f'output/sleemory_localiser_vox/dnn_feature_maps/best_feature_maps/sub_{args.sub}/{fmaps_fname}')

# Load fmaps and labels
fmaps = fmaps_data['fmaps'] # (img, 'num_token', num_feat)
if args.networks == 'BLIP-2': # Need to select token or mean pooling
    fmaps = np.mean(fmaps, axis=1)
fmaps_labels = fmaps_data['imgs_all']
del fmaps_data

### Load retrieval image names 
img_dir = f'/home/simon/Documents/gitrepos/shannon_encodingmodelsEEG/dataset//sleemory_retrieval/image_set/'
img_list = os.listdir(img_dir)

retrieval_fmaps = []
for img in img_list:
	fmaps_idx = np.where(fmaps_labels == img)
	retrieval_fmaps.append(fmaps[fmaps_idx])
retrieval_fmaps = np.asarray(retrieval_fmaps)

# Prediction 
pred_eeg = reg.predict(retrieval_fmaps)
pred_eeg = pred_eeg.reshape(pred_eeg.shape[0], 58, 363) # (img, ch, time)

# Save
save_dir = f'output/sleemory_retrieval_vox/pred_eeg/sub-{args.sub}/'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
pred_eeg_fname = f'{args.networks}_pred_eeg.mat'
print(pred_eeg_fname)
scipy.io.savemat(f'{save_dir}/{pred_eeg_fname}', {'pred_eeg': pred_eeg,
                                                  'imgs_all': img_list}) 