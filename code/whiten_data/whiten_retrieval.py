import os
import scipy
import numpy as np
from func import mvnn


print('')
print(f'>>> Whiten sleemory retrieval EEG data <<<')
print('')


# List of imgs names
imgs_names = os.listdir('dataset/sleemory_retrieval/image_set')
imgs_names = [name[:-4] for name in imgs_names]


for sub in range(2, 3):
    if sub == 17:
        pass
    else:
        # Load the test EEG data
        eeg_dir = 'dataset/sleemory_retrieval/preprocessed_data'
        data = scipy.io.loadmat(os.path.join(eeg_dir,
                                f'sleemory_retrieval_dataset_sub-{sub:03d}.mat'))
        eegs = data['ERP_all']
        imgs = data['imgs_all']

        sorted_eeg_all = []
        for ses in range(2):
            
            eegs_part = eegs[:, ses][0]
            imgs_part = imgs[:, ses][0][:,0]
            
            # Classify eeg data according to imgs names
            sorted_eegs, sorted_idx = [], []
            for i, name in enumerate(imgs_names):
                mask = imgs_part == name
                eeg = eegs_part[mask]
                print('Original test_eeg_data shape (img, ch, time)', eeg.shape)
                sorted_eegs.append(eeg)
            sorted_idx = np.asarray(sorted_idx)
                           
            # Whiten the data
            whitened_data = mvnn(sorted_eegs)
            sorted_eeg_all.append(whitened_data)
            
        # Save the whitened eeg data
        save_dir = f'output/sleemory_retrieval/whiten_eeg'
        if os.path.isdir(save_dir) == False:
            os.makedirs(save_dir)
            
        save_dict = {'whitened_data': sorted_eeg_all, 'imgs_all': imgs_names}
        np.save(os.path.join(save_dir, f'whiten_test_eeg_sub-{sub:03d}'), save_dict)
        # scipy.io.savemat(os.path.join(save_dir, f'whiten_test_eeg.mat'), save_dict)