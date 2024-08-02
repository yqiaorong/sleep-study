import os
import scipy
import numpy as np
from func import mvnn

dataset = 'retrieval'
print('')
print(f'>>> Whiten sleemory {dataset} EEG data (original order) <<<')
print('')


# List of imgs names
imgs_names = os.listdir(f'dataset/sleemory_{dataset}/image_set')
imgs_names = [name[:-4] for name in imgs_names]

for sub in range(2, 27):
    if sub == 17:
        pass
    else:
        # Load the test EEG data
        eeg_dir = f'dataset/sleemory_{dataset}/preprocessed_data'
        data = scipy.io.loadmat(f'{eeg_dir}/sleemory_retrieval_dataset_sub-{sub:03d}.mat')
        eegs_sub = data['ERP_all'] # (1, 2)
        imgs_sub = data['imgs_all'] # (1, 2)
        del data
        
        sorted_eeg_all = []
        for ses in range(2):
            eegs_ses = eegs_sub[:, ses][0]
            imgs_ses = imgs_sub[:, ses][0][:,0]
            
            # Classify eeg data according to imgs names
            whitened_eegs_re = np.empty(eegs_ses.shape) # (100, num_ch, num_t)

            true_indices, eegs = [], []
            for i, name in enumerate(imgs_names):
                mask = imgs_ses == name
                
                # Mark the index 
                true_idx = np.where(mask)[0]
                true_indices.append(true_idx)
                
                # Extract the eeg
                eeg = eegs_ses[mask] # (25, num_ch, num_t)
                eegs.append(eeg)
                
            # Whiten the data
            whitened_eegs = mvnn(eegs)
                
            # Assign the whitened data to final whitened data with original order
            for i, name in enumerate(imgs_names):
                whitened_eegs_re[true_indices[i]] = whitened_eegs[i]
            sorted_eeg_all.append(whitened_eegs_re)
             
        sorted_eeg_all = np.array(sorted_eeg_all)
        print(sorted_eeg_all.shape)
        # Save the whitened eeg data
        save_dir = f'output/sleemory_{dataset}/whiten_eeg_original'
        if os.path.isdir(save_dir) == False:
            os.makedirs(save_dir)
            
        save_dict = {'whitened_data': sorted_eeg_all, 'imgs_all': imgs_sub}
        scipy.io.savemat(os.path.join(save_dir, f'whiten_test_eeg_sub-{sub:03d}.mat'), save_dict)