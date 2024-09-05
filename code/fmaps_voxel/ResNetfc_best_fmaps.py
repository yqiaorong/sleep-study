import os
import argparse
import numpy as np
import scipy
import mat73
from sklearn.feature_selection import SelectKBest, f_regression

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--num_feat',default=1000, type=int)
parser.add_argument('--sub',     default=2,    type=int)
args = parser.parse_args()

print('')
print(f'>>> Sleemory images best feature maps (ResNet-fc) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')

networks = 'ResNet-fc'
# =============================================================================
# Load eeg data
# =============================================================================   

# Load localiser EEG
eeg_dir = '/home/simon/Documents/gitrepos/shannon_encodingmodelsEEG/dataset/sleemory_localiser/preprocessed_data'
eeg_fname = f'sub-{args.sub:03d}_task-localiser_source_data'

eeg_data = mat73.loadmat(os.path.join(eeg_dir, eeg_fname+'.mat'))
eeg = eeg_data['sub_eeg_loc']['eeg']

eeg_labels = eeg_data['sub_eeg_loc']['images']
eeg_labels = [s[0].split('.')[0] for s in eeg_labels]
eeg_labels = np.asarray(eeg_labels)
    
# Average across time to shape (img, ch,)
eeg = np.mean(eeg, -1)
# Average across channel to shape (img,)
eeg = np.mean(eeg, -1)
print('(localiser) EEG data shape (img,)', eeg.shape)

# =============================================================================
# Load fmaps
# =============================================================================

# Load fmaps
fmaps_data = scipy.io.loadmat(f'dataset/sleemory_localiser/dnn_feature_maps/full_feature_maps/{networks}/gptneo_fmaps.mat')

fmaps_labels = fmaps_data['imgs_all']
fmaps = fmaps_data['fmaps']
print(f'fmaps all shape: {fmaps.shape}')
del fmaps_data

# Select the matching fmaps
select_fmaps = np.empty((eeg.shape[0], fmaps.shape[1]))
select_labels, select_capts = [], []
for ieeg_label, eeg_label in enumerate(eeg_labels):
    fmaps_idx = np.where(fmaps_labels == eeg_label)[0]
    select_fmaps[ieeg_label] = fmaps[fmaps_idx]
    select_labels.append(fmaps_labels[fmaps_idx])

select_labels = np.squeeze(np.asarray(select_labels))

print(f'fmaps selected shape: {select_fmaps.shape}') 
print(f'fmaps labels selected shape: {select_labels.shape}') 
del fmaps

# =============================================================================
# Feature selection
# =============================================================================
    
# Build the feature selection model
feature_selection = SelectKBest(f_regression, 
                                k=args.num_feat).fit(select_fmaps, eeg)
# Select the best features
select_fmaps = feature_selection.transform(select_fmaps)
print(f'The final fmaps selected has shape {select_fmaps.shape}')

# =============================================================================
# Save new features
# =============================================================================

# Save dir
train_save_dir = f'output/sleemory_localiser_vox/dnn_feature_maps/best_feature_maps/sub_{args.sub}/'
if os.path.isdir(train_save_dir) == False:
    os.makedirs(train_save_dir)

# Save
best_fmaps_fname = f'{networks}-best-{args.num_feat}_fmaps.mat'
print(best_fmaps_fname)

scipy.io.savemat(f'{train_save_dir}/{best_fmaps_fname}', {'fmaps': select_fmaps,
                                                          'imgs_all': np.char.rstrip(select_labels)}) 

# Save the model
import pickle
model_dir = f'output/sleemory_localiser_vox/model/best_feat_model/sub-{args.sub}/'
if os.path.isdir(model_dir) == False:
	os.makedirs(model_dir)
reg_model_fname = f'{networks}_best_feat_model.pkl'
print(reg_model_fname)
pickle.dump(feature_selection, open(f'{model_dir}/{reg_model_fname}','wb'))