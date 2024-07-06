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
args = parser.parse_args()

print('')
print(f'>>> Predict EEG from the encoding model <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
 
 

# Load encoding model
reg = pickle.load(open(f'dataset/sleemory_localiser/model/reg_model/{args.networks}_reg_model.pkl', 'rb'))


# Pred test EEG from test fmaps
dnn_test_dir = os.path.join('dataset/sleemory_retrieval/dnn_feature_maps')
dnn_fmaps_test = scipy.io.loadmat(f'{dnn_test_dir}/{args.networks}_fmaps.mat')
# Load fmaps
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
scipy.io.savemat(f'{save_dir}/{args.networks}_pred_eeg.mat', {'pred_eeg': pred_eeg}) 