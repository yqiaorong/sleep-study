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
# num_feat = -1 means using all features, If it's Alexnet, num_feat cannot be -1
parser.add_argument('--whiten',  default=False, type=bool) # This indicates the reg model
args = parser.parse_args()

print('')
print(f'>>> Predict EEG from the encoding model <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')
 
# Setup
if args.whiten == False:
    whiten = ''
else:
    whiten = 'whiten_'
    
if args.num_feat == -1:
    best_feat_cond = ''
    fmaps_whiten = ''
else:
    best_feat_cond = f'-best-{args.num_feat}'
    fmaps_whiten = whiten
    
# Load encoding model
reg_model_fname = f'{args.networks}{best_feat_cond}_{whiten}reg_model.pkl'
print(reg_model_fname)
reg = pickle.load(open(f'dataset/sleemory_localiser/model/reg_model/{reg_model_fname}', 'rb'))


# Load test fmaps
dnn_fmaps_fname = f'{args.networks}{best_feat_cond}_{fmaps_whiten}fmaps.mat'
print(dnn_fmaps_fname)
dnn_fmaps_test = scipy.io.loadmat(f'dataset/sleemory_retrieval/dnn_feature_maps/{dnn_fmaps_fname}')
test_fmap = dnn_fmaps_test['fmaps'] # (img, 'num_token', num_feat)
if args.networks == 'BLIP-2': # Need to select token or mean pooling
    test_fmap = np.mean(test_fmap, axis=1)
    
# Prediction 
pred_eeg = reg.predict(test_fmap)
pred_eeg = pred_eeg.reshape(pred_eeg.shape[0], 58, 363) # (img, ch, time)

# Save
save_dir = 'output/sleemory_retrieval/pred_eeg'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
pred_eeg_fname = f'{args.networks}{best_feat_cond}_{whiten}pred_eeg.mat'
print(pred_eeg_fname)
scipy.io.savemat(f'{save_dir}/{pred_eeg_fname}', {'pred_eeg': pred_eeg}) 