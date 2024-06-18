import numpy as np
import os
from func import plot2D

# Input
sub_list = [9]
ses_list = [0, 1]
num_feat = 1000
layer = 'conv5'
whiten = True

for sub in sub_list:
    for ses in ses_list:
        
        # Load the data
        load_dir = f'output/sleemory_retrieval/enc_acc/sub-{sub:03d}/ses-{ses}/{num_feat}feats_whiten{whiten}'
        load_list = os.listdir(load_dir)

        # Average the data
        avg_data = []
        for fname in load_list:
            if fname.endswith(f'{layer}_enc_acc.npy'):
                data = np.load(os.path.join(load_dir, fname), allow_pickle=True)
                data = np.mean(data, axis=0)
                avg_data.append(data)
        avg_data = np.mean(avg_data, axis=0)

        # Plot the data
        save_dir = f'output/sleemory_retrieval/enc_acc/sub-{sub:03d}/ses-{ses}/{num_feat}feats_whiten{whiten}'
        if os.path.isdir(save_dir) == False:
            os.makedirs(save_dir)  
        save_path = os.path.join(save_dir, f'avg_{layer}_enc_acc')

        plot2D(avg_data, (10, 5), [-0.25, 1], [0, 2.5], 
              'Pred EEG time / s', 'Test EEG time / s', save_path)