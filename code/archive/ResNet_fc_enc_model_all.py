import os
import numpy as np
from tqdm import tqdm

for vox_idx in tqdm(range(5, 3294)):
   os.system(f'python3 code/vox/ResNet_fc_enc_model.py --vox_idx {vox_idx}')

# Concatenate all pred eeg
load_dir = 'output/sleemory_retrieval_vox/pred_eeg_voxelwise/'

tot_pred_eeg = []
for vox_idx in range(3294):
    fname = f'ResNet_fc_vox_{vox_idx:04d}.npy'
    data = np.load(load_dir+fname, allow_pickle=True).item()

    pred_eeg = data['pred_eeg']
    print(pred_eeg.shape)
    tot_pred_eeg.append(pred_eeg)
    
    if vox_idx == 0:
        flabels = data['imgs_all']
tot_pred_eeg = np.array(tot_pred_eeg)
tot_pred_eeg = tot_pred_eeg.swapaxes(0,1)
print(tot_pred_eeg.shape)
print(flabels)
np.save(load_dir+f'ResNet_fc_pred_eeg', {'pred_eeg': tot_pred_eeg,
										 'imgs_all': flabels})