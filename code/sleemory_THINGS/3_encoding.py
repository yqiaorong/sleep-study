import os
import scipy
import pickle
import argparse
import numpy as np
from tqdm import tqdm

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None,       type=str)
parser.add_argument('--pretrained', default=True,    type=bool)
parser.add_argument('--layer_name', default='conv5', type=str)

parser.add_argument('--num_feat', default=1000,      type=int)
parser.add_argument('--z_score', default=True,       type=bool)
args = parser.parse_args()

print('')
print(f'>>> Encoding on sleemory based on THINGS <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')

# =============================================================================
# Load the test feature maps after feature selection
# =============================================================================

fmaps_path = f'dataset/{args.dataset}/dnn_feature_maps/new_feature_maps_{args.num_feat}.npy'
best_feat_test = np.load(fmaps_path, allow_pickle=True)

print(f'The new fmaps shape (img, feat) {best_feat_test.shape}')

# =============================================================================
# Load the encoding model and make predictions
# =============================================================================

enc_model_path = f'dataset/THINGS_EEG2/model'
reg = pickle.load(open(os.path.join(enc_model_path, 
                                    f'reg_model_{args.num_feat}_sleemory.pkl'), 
                       'rb'))

# Predict the EEG data 
pred_eeg = reg.predict(best_feat_test)

# =============================================================================
# Drop extra img cond in pred eeg
# =============================================================================

# Get img stimuli list
raw_eeg = scipy.io.loadmat(f'dataset/{args.dataset}/preprocessed_data'+
                            '/sleemory_localiser_dataset.mat')
imgs_all = raw_eeg['imgs_all']
del raw_eeg

# Get number of channels and time points
eeg_path = f'output/{args.dataset}_THINGS/test_eeg/whiten_test_eeg.npy'
load_eeg = np.load(eeg_path, allow_pickle=True).item()
eeg = load_eeg['test_eeg2']
num_ch, t_sleemory = eeg.shape[1], eeg.shape[2]

# Reshape the predicted EEG data
t_THINGS = 250
pred_eeg = np.reshape(pred_eeg, (pred_eeg.shape[0], num_ch, t_THINGS))
print('Original pred_eeg_data shape (img, ch, time)', pred_eeg.shape)

# Find indices in A that match the first element of B
unique_imgs = np.unique(imgs_all)
unique_imgs = [item for img in unique_imgs for item in img]

image_set_list = os.listdir(f'dataset/{args.dataset}/image_set')

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
# Save the predict eeg data
# =============================================================================

# Create the saving directory
save_dir = f'output/{args.dataset}_THINGS/test_eeg'
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)
    
np.save(os.path.join(save_dir, f'z{args.z_score}_{args.num_feat}feat_pred_eeg'), pred_eeg)