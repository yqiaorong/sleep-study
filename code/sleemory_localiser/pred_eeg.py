import pickle
import os
import scipy

# Load encoding model
layer_name = 'CLIP'
reg = pickle.load(open(f'dataset/sleemory_localiser/model/reg_model/{layer_name}_reg_model.pkl', 'rb'))


# Pred test EEG from test fmaps
dnn_test_dir = os.path.join('dataset/sleemory_retrieval/dnn_feature_maps')
dnn_fmaps_test = scipy.io.loadmat(f'{dnn_test_dir}/CLIP_fmaps.mat')
# Load fmaps
test_fmap = dnn_fmaps_test['fmaps'] # (img, feats)
# Prediction 
pred_eeg = reg.predict(test_fmap)
pred_eeg = pred_eeg.reshape(pred_eeg.shape[0], 58, 363) # (img, ch, time)
# Save
save_dir = 'output/sleemory_retrieval/pred_eeg'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
scipy.io.savemat(f'{save_dir}/{layer_name}_pred_eeg.mat', {'pred_eeg': pred_eeg}) 