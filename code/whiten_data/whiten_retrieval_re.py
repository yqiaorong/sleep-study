import os
import scipy
import numpy as np
from func import mvnn


print('')
print(f'>>> Whiten sleemory retrieval EEG data (original order) <<<')
print('')

dataset = 'retrieval'


# List of imgs names
imgs_names = os.listdir('dataset/sleemory_retrieval/image_set')
imgs_names = [name[:-4] for name in imgs_names]


for sub in range(2, 5):
    if sub == 17:
        pass
    else:
        # Load the test EEG data
        eeg_dir = f'dataset/sleemory_{dataset}/preprocessed_data'
        data = scipy.io.loadmat(os.path.join(eeg_dir,
                                f'sleemory_retrieval_dataset_sub-{sub:03d}.mat'))
        eegs_sub = data['ERP_all'] # (1, 2)
        imgs_sub = data['imgs_all'] # (1, 2)
        del data

        sorted_eeg_all = [] # final data of two sessions
        for ses in range(2):
            
            eegs_ses = eegs_sub[:, ses][0]
            imgs_ses = imgs_sub[:, ses][0][:,0]
            
            # Classify eeg data according to imgs names
            whitened_data_re = np.empty(eegs_ses.shape) # (100, num_ch, num_t)
            
            for i, name in enumerate(imgs_names):
                mask = imgs_ses == name
                
                # Mark the index 
                true_idx = np.where(mask)[0]
                
                # Extract the eeg
                eeg = eegs_ses[mask] # (25, num_ch, num_t)
                
                # Whiten the data
                whitened_data = mvnn([eeg])
                whitened_data = np.squeeze(np.asarray(whitened_data)) # (25, num_ch, num_t)
                
                # Assign the whitened data to final whitened data with original order
                whitened_data_re[true_idx] = whitened_data
                del whitened_data
                
            # Append two sessions data
            sorted_eeg_all.append(whitened_data_re)
            
        # Save the whitened eeg data
        save_dir = f'output/sleemory_{dataset}/whiten_eeg_original'
        if os.path.isdir(save_dir) == False:
            os.makedirs(save_dir)
            
        save_dict = {'whitened_data': sorted_eeg_all, 'imgs_all': imgs_sub}
        np.save(os.path.join(save_dir, f'whiten_test_eeg_sub-{sub:03d}'), save_dict)
        scipy.io.savemat(os.path.join(save_dir, f'whiten_test_eeg_sub-{sub:03d}.mat'), save_dict)