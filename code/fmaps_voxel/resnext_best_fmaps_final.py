import os
import argparse
import numpy as np
import scipy
from tqdm import tqdm
from sklearn.feature_selection import SelectKBest, f_regression
import mat73

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=2, type=int)
parser.add_argument('--new_num_feat', default=1000, type=int)
parser.add_argument('--old_num_feat', default=3000, type=int)
args = parser.parse_args()

print('')
print(f'>>> Feature selection of sleemory images feature maps (resnext) final round <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')

# =============================================================================
# Load eeg data
# =============================================================================

# Load localiser EEG
eeg_dir = '/home/simon/Documents/gitrepos/shannon_encodingmodelsEEG/dataset/sleemory_localiser/preprocessed_data'

eeg_fname = f'sub-{args.sub:03d}_task-localiser_source_data'
eeg_data = mat73.loadmat(os.path.join(eeg_dir, eeg_fname+'.mat'))
eeg = eeg_data['sub_eeg_loc']['eeg']

# Average across time to shape (img, ch,)
eeg = np.mean(eeg, -1)
# Average across channel to shape (img,)
eeg = np.mean(eeg, -1)
print('Training (localiser) EEG data shape (img,)', eeg.shape)

# =============================================================================
# Load feature maps
# =============================================================================

fmaps_dir = f'output/sleemory_localiser_vox/dnn_feature_maps/best_feature_maps/sub_{args.sub}/ResNet'

# file list
layer_start_indices = range(0, 287, 50)
file_list = [f'ResNet-best-{args.old_num_feat}-{idx}_fmaps.mat' for idx in layer_start_indices]

for ichunk, chunk in enumerate(tqdm(file_list, desc='chunk file')):
    fmaps = scipy.io.loadmat(fmaps_dir+chunk)
    fmaps_labels = fmaps['imgs_all']
    
    # Concatenate layers in chunk file
    for ilayer, layer in enumerate(list(fmaps.keys())[3:]):
        if ilayer == 0:
            fmaps_chunk = fmaps[layer]
        else:
            fmaps_chunk = np.concatenate((fmaps_chunk, fmaps[layer]), axis=1)
    print(f'train fmaps chunk shape: {fmaps_chunk.shape}')
    del fmaps
    
    # Concatenate chunk files
    if ichunk == 0:
        fmaps_all = fmaps_chunk
    else:
        fmaps_all = np.concatenate((fmaps_all, fmaps_chunk), axis=1)
    del fmaps_chunk
print(f'The fmaps all shape: {fmaps_all.shape}')  
print('')

# =============================================================================
# Feature selection
# =============================================================================
    
# Build the feature selection model
feature_selection = SelectKBest(f_regression, 
                                k=args.new_num_feat).fit(fmaps_all, eeg)

# Select the best features
train_fmaps_all = feature_selection.transform(fmaps_all)
print(f'The final fmaps has shape {train_fmaps_all.shape}')

# =============================================================================
# Save new features
# =============================================================================

# Save dir
train_save_dir = f'output/sleemory_localiser_vox/dnn_feature_maps/best_feature_maps/sub_{args.sub}/'
if os.path.isdir(train_save_dir) == False:
    os.makedirs(train_save_dir)

# Save
best_fmaps_fname = f'ResNet-best-{args.new_num_feat}_fmaps.mat'
print(best_fmaps_fname)

scipy.io.savemat(f'{train_save_dir}/{best_fmaps_fname}', {'fmaps': train_fmaps_all,
                                                          'imgs_all': fmaps_labels}) 