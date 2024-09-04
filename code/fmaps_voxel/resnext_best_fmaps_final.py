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

fname = f'sub-{args.sub:03d}_task-localiser_source_data'
data = mat73.loadmat(os.path.join(eeg_dir, fname+'.mat'))
train_eeg = data['sub_eeg_loc']['eeg']

# Average across time to shape (img, ch,)
train_eeg = np.mean(train_eeg, -1)
# Average across channel to shape (img,)
train_eeg = np.mean(train_eeg, -1)
print('Training (localiser) EEG data shape (img,)', train_eeg.shape)

# =============================================================================
# Load training feature maps
# =============================================================================

train_fmaps_dir = f'output/sleemory_localiser_vox/dnn_feature_maps/best_feature_maps/sub_{args.sub}/ResNet'

# file list
layer_start_indices = range(0, 287, 50)
file_list = [f'ResNet-best-{args.old_num_feat}-{idx}_fmaps.mat' for idx in layer_start_indices]

for ichunk, chunk in enumerate(tqdm(file_list, desc='train chunk file')):
    train_fmaps = scipy.io.loadmat(train_fmaps_dir+chunk)
    
    # Concatenate layers in chunk file
    for ilayer, layer in enumerate(list(train_fmaps.keys())[3:]):
        if ilayer == 0:
            train_fmaps_chunk = train_fmaps[layer]
        else:
            train_fmaps_chunk = np.concatenate((train_fmaps_chunk, train_fmaps[layer]), axis=1)
    print(f'train fmaps chunk shape: {train_fmaps_chunk.shape}')
    del train_fmaps
    
    # Concatenate chunk files
    if ichunk == 0:
        train_fmaps_all = train_fmaps_chunk
    else:
        train_fmaps_all = np.concatenate((train_fmaps_all, train_fmaps_chunk), axis=1)
    del train_fmaps_chunk
print(f'train fmaps all shape: {train_fmaps_all.shape}')  
print('')

# =============================================================================
# Feature selection
# =============================================================================
    
# Build the feature selection model
feature_selection = SelectKBest(f_regression, 
                                k=args.new_num_feat).fit(train_fmaps_all, train_eeg)

# Select the best features
train_fmaps_all = feature_selection.transform(train_fmaps_all)
print(f'The final train fmaps has shape {train_fmaps_all.shape}')

# =============================================================================
# Save new features
# =============================================================================

# Save dir
train_save_dir = f'output/sleemory_localiser_vox/dnn_feature_maps/'
if os.path.isdir(train_save_dir) == False:
    os.makedirs(train_save_dir)

# Save
best_fmaps_fname = f'ResNet-best-{args.new_num_feat}_fmaps.mat'
print(best_fmaps_fname)

# Load training fmaps
img_dir = f'dataset/sleemory_localiser/dnn_feature_maps/full_feature_maps/ResNet/'
img_list = os.listdir(img_dir)
img_list = [img.split('.')[0] for img in img_list]
print(img_list)
scipy.io.savemat(f'{train_save_dir}/{best_fmaps_fname}', {'fmaps': train_fmaps_all,
                                                          'imgs_all': img_list}) 