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
parser.add_argument('--sub',     default=None,    type=int)
args = parser.parse_args()

networks = 'gptneo'

print('')
print(f'>>> Sleemory images best feature maps ({networks}) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')

# =============================================================================
# Load eeg data (checked)
# =============================================================================   

# Load localiser EEG
eeg_dir = '/home/simon/Documents/gitrepos/shannon_encodingmodelsEEG/dataset/sleemory_localiser/preprocessed_data'
eeg_fname = f'sub-{args.sub:03d}_task-localiser_source_data'

eeg_data = mat73.loadmat(os.path.join(eeg_dir, eeg_fname+'.mat'))
eeg = eeg_data['sub_eeg_loc']['eeg']

eeg_labels = eeg_data['sub_eeg_loc']['images']
eeg_labels = [s[0] for s in eeg_labels]
eeg_labels = np.asarray(eeg_labels)

    
# Average across time to shape (img, ch,)
eeg = np.mean(eeg, -1)
# Average across channel to shape (img,)
eeg = np.mean(eeg, -1)
print('(localiser) EEG data shape (img,)', eeg.shape)

# =============================================================================
# Load fmaps (checked)
# =============================================================================

fmaps_data = scipy.io.loadmat(f'dataset/sleemory_localiser/dnn_feature_maps/full_feature_maps/{networks}/{networks}_fmaps.mat')

fmaps_labels = fmaps_data['imgs_all']
fmaps_labels = [np.char.rstrip(s) for s in fmaps_labels] # remove extra spacing in strings
fmaps_labels = np.array(fmaps_labels)

fmaps_capts  = fmaps_data['captions']
for ilayer, layer in enumerate(fmaps_data.keys()):
    print(layer)
    if layer.startswith('layer'):
        print(fmaps_data[layer].shape, 'take')
        if layer == 'layer_0_embeddings':
            fmaps = fmaps_data[layer]
        else:
            fmaps = np.concatenate((fmaps, fmaps_data[layer]), axis=1)
print(f'fmaps all shape: {fmaps.shape}')
del fmaps_data

# =============================================================================
# Select the matching fmaps (checked)
# =============================================================================

select_fmaps = np.empty((eeg.shape[0], fmaps.shape[1]))
select_labels, select_capts = [], []
for ieeg_label, eeg_label in enumerate(eeg_labels):
    print(eeg_label)
    print(np.where(fmaps_labels == eeg_label))
    fmaps_idx = np.where(fmaps_labels == eeg_label)[0]
    print(f'fmaps idx: {fmaps_idx}')
    
    if eeg_label == fmaps_labels[fmaps_idx]:
        print('fmaps idx correct')
        select_fmaps[ieeg_label] = fmaps[fmaps_idx]
        select_labels.append(fmaps_labels[fmaps_idx])
        select_capts.append(fmaps_capts[fmaps_idx])
    else:
        print('fmaps idx incorrect')
    print('')

print(f'fmaps selected shape: {select_fmaps.shape}') 
print(f'fmaps labels selected length: {len(select_labels)}') 
print(f'fmaps capts selected length: {len(select_capts)}') 
del fmaps, fmaps_labels, fmaps_capts
print(select_labels)
print(select_capts)

# =============================================================================
# Feature selection
# =============================================================================
    
# Build the feature selection model
feature_selection = SelectKBest(f_regression, 
                                k=args.num_feat).fit(select_fmaps, eeg)
# Select the best features
select_fmaps = feature_selection.transform(select_fmaps)
print(f'The final fmaps selected shape {select_fmaps.shape}')

# =============================================================================
# Save the best features and the best feature models
# =============================================================================

# Save dir
train_save_dir = f'output/sleemory_localiser_vox/dnn_feature_maps/best_feature_maps/sub_{args.sub}/'
if os.path.isdir(train_save_dir) == False:
    os.makedirs(train_save_dir)

# Save
best_fmaps_fname = f'{networks}-best-{args.num_feat}_fmaps.mat'
print(best_fmaps_fname)
scipy.io.savemat(f'{train_save_dir}/{best_fmaps_fname}', {'fmaps': select_fmaps,
                                                          'imgs_all': select_labels,
                                                          'captions': select_capts}) 

# Save the model
import pickle
model_dir = f'output/sleemory_localiser_vox/model/best_feat_model/sub-{args.sub}/'
if os.path.isdir(model_dir) == False:
	os.makedirs(model_dir)
reg_model_fname = f'{networks}_best_feat_model.pkl'
print(reg_model_fname)
pickle.dump(feature_selection, open(f'{model_dir}/{reg_model_fname}','wb'))

# =============================================================================
# Apply the best feature models on retrieval fmaps (checked)
# =============================================================================

# Load retrieval feature maps
test_fmaps_data = scipy.io.loadmat(f'dataset/sleemory_retrieval/dnn_feature_maps/full_feature_maps/{networks}/{networks}_fmaps.mat')

test_fmaps_labels = test_fmaps_data['imgs_all']
test_fmaps_labels = [np.char.rstrip(s) for s in test_fmaps_labels] # remove extra spacing in strings
test_fmaps_labels = np.array(test_fmaps_labels)
print(test_fmaps_labels)

test_fmaps_capts  = test_fmaps_data['captions']
print(test_fmaps_labels)

for ilayer, layer in enumerate(test_fmaps_data.keys()):
    print(layer)
    if layer.startswith('layer'):
        print(test_fmaps_data[layer].shape, 'take')
        if layer == 'layer_0_embeddings':
            test_fmaps = test_fmaps_data[layer]
        else:
            test_fmaps = np.concatenate((test_fmaps, test_fmaps_data[layer]), axis=1)
print(f'The test fmaps shape: {test_fmaps.shape}')
del test_fmaps_data

# Apply the best feature models 
test_fmaps = feature_selection.transform(test_fmaps)
print(f'The final test fmaps selected shape {test_fmaps.shape}')

# Save dir
test_save_dir = f'output/sleemory_retrieval_vox/dnn_feature_maps/best_feature_maps/sub_{args.sub}/'
if os.path.isdir(test_save_dir) == False:
    os.makedirs(test_save_dir)

# Save the test features
print(best_fmaps_fname)
scipy.io.savemat(f'{test_save_dir}/{best_fmaps_fname}', {'fmaps': test_fmaps,
                                                          'imgs_all': test_fmaps_labels,
                                                          'captions': test_fmaps_capts}) 