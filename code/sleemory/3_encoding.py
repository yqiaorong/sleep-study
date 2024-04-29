import os
import scipy
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from func import mvnn_mean, mvnn

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--layer_name', default='conv5', type=str)
parser.add_argument('--num_feat', default=1000, type=int)
parser.add_argument('--z_score', default=True, type=bool)
args = parser.parse_args()

print('')
print(f'>>> Encoding on sleemory <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')

# =============================================================================
# Load the test feature maps after feature selection
# =============================================================================

fmaps_path = f'dataset/temp_sleemory/dnn_feature_maps/new_feature_maps_{args.num_feat}.npy'
best_feat_test = np.load(fmaps_path, allow_pickle=True)

print(f'The new fmaps shape (img, feat) {best_feat_test.shape}')

# =============================================================================
# Load the encoding model
# =============================================================================

enc_model_path = f'dataset/THINGS_EEG2/model'
reg = pickle.load(open(os.path.join(enc_model_path, 
                                    f'reg_model_{args.num_feat}_sleemory.pkl'), 
                       'rb'))

# =============================================================================
# Predict EEG from fmaps
# =============================================================================

# Load the test EEG data directory
eeg_dir = os.path.join('dataset', 'temp_sleemory', 'preprocessed_data')
data = scipy.io.loadmat(os.path.join(eeg_dir,'sleemory_localiser_dataset.mat'))
prepr_data = data['ERP_all']
imgs_all = data['imgs_all']
ch_names = []
for ch in data['channel_names'][:,0]:
    ch_names.append(ch[0])
times = data['time']
# set time
t_sleemory, t_THINGS = times.shape[1], 250
del data

# Drop the extra channel 'Fpz' and 'Fz':
idx_Fz, idx_Fpz = ch_names.index('Fz'), ch_names.index('Fpz')
prepr_data = np.delete(prepr_data, [idx_Fz, idx_Fpz], axis=1)
print('Original test_eeg_data shape (img, ch, time)', prepr_data.shape)
# set channels
num_ch = len(ch_names)-2

# Predict the EEG data 
pred_eeg = reg.predict(best_feat_test)
# Reshape the predicted EEG data
pred_eeg = np.reshape(pred_eeg, (pred_eeg.shape[0], num_ch, t_THINGS))
print('Original pred_eeg_data shape (img, ch, time)', pred_eeg.shape)

# =============================================================================
# Drop extra img cond in pred eeg
# =============================================================================

# Find indices in A that match the first element of B
unique_imgs = np.unique(imgs_all)
unique_imgs = [item for img in unique_imgs for item in img]

image_set_list = os.listdir('dataset/temp_sleemory/image_set')

# There is one extra fmap which should be dropped from pred eeg
exclude_img = list(set(image_set_list)-set(unique_imgs))
exclude_idx_in_img_set = image_set_list.index(exclude_img[0])
print(f"The dropped img cond's idx: {exclude_idx_in_img_set}")
pred_eeg = np.delete(pred_eeg, exclude_idx_in_img_set, axis=0)
print("One img cond is dropped from pred eeg since it's absent in test eeg.")

# cross check the order of unique images
for i, j in zip(unique_imgs, image_set_list[1:]):
    if i == j:
        pass
    else:
        print('The order of img is unmatched')
        break
    
# =============================================================================
# Sort test EEG data
# =============================================================================

# Create the saving directory
save_dir = f'output/sleemory/test_eeg'
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)
    
# =============================================================================
# Categorize the preprocessed data
# =============================================================================

# Sort the test eeg data
test_eeg = np.empty((len(unique_imgs), num_ch, t_sleemory)) # storing mean EEG for each img
tot_test_eeg = [] # storing all EEG for each img
# Iterate over images
for idx, img in enumerate(tqdm(unique_imgs, desc='Average test eeg across unique images')):
    img_indices = np.where(imgs_all == img)[0]
    # select corresponding prepr data
    select_data = prepr_data[img_indices]
    # Append data
    tot_test_eeg.append(select_data)
    
    # Average across the same images
    select_data = np.mean(select_data, 0)
    test_eeg[idx] = select_data

# =============================================================================
# Z score the data
# =============================================================================

if args.z_score == True:
    test_eeg = mvnn_mean(test_eeg)
    tot_test_eeg = mvnn(tot_test_eeg)
else:
    pass

# Average z scored total test eeg data
test_eeg2 = np.empty(test_eeg.shape)
for i, data in enumerate(tot_test_eeg):
    new_data = np.mean(data, axis=0)
    test_eeg2[i] = new_data
del tot_test_eeg

# =============================================================================
# Save the test eeg data
# =============================================================================

save_dict = {'test_eeg': test_eeg, 'test_eeg2': test_eeg2, 'pred_eeg': pred_eeg}
np.save(os.path.join(save_dir, f'z{args.z_score}_{args.num_feat}feat'), save_dict)