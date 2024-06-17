import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt

whiten = True
sub = 2

data = np.load(f'output/sleemory_retrieval/whiten_eeg/whiten_test_eeg_sub-{sub:3d}.npy', allow_pickle=True).item()
eegs = data['whitened_data']
imgs_names = data['imgs_all']


# Load predicted test eeg from the encoding model
num_feat = 1000
pred_eeg_dir = f'output/sleemory_retrieval/test_pred_eeg/pred_eeg_with_{num_feat}feats.npy'
pred_eeg_all = np.load(pred_eeg_dir, allow_pickle=True).item() # dict of 8 layers
layer = 'conv5'
pred_eeg = pred_eeg_all[layer]
pred_time = pred_eeg.shape[2]


# PM and AM sessions
for ses in range(2):
    eegs_part = eegs[ses]
    
    for i, name in enumerate(imgs_names):
        
        eeg = eegs_part[i]
        num_trial = eeg.shape[0]
        test_time = eeg.shape[2]
        
        # Correlations
        enc_acc = np.empty((num_trial, pred_time, test_time))
        
        for stimuli_idx in tqdm(range(num_trial), desc='Iteration over stimuli'):
            for t_test in range(test_time):
                test_val = pd.Series(eeg[stimuli_idx, :, t_test])
                for t_pred in range(pred_time):
                    pred_val = pd.Series(pred_eeg[i, :, t_pred])
                    enc_acc[stimuli_idx, t_pred, t_test] = test_val.corr(pred_val)
                    
        # Save data
        save_dir = f'output/sleemory_retrieval/enc_acc/sub-{sub:03d}/ses-{ses}/{num_feat}feats_whiten{whiten}'
        if os.path.isdir(save_dir) == False:
            os.makedirs(save_dir)
            
        np.save(os.path.join(save_dir, f'{layer}_enc_acc'), enc_acc)
        
        # Average the results over stimuli
        avg_enc_acc = np.mean(enc_acc, axis=0)
        
        # Plot all 2D results of method 1
        fig = plt.figure(figsize=(6, 5))
        im = plt.imshow(avg_enc_acc, cmap='viridis',
        				extent=[0, 2.5, -0.25, 1], 
                        origin='lower', aspect='auto')
        cbar = plt.colorbar(im)
        cbar.set_label('Values')

        # Plot borders
        plt.plot([0, 2.5], [0,0], 'k--', lw=0.4)
        plt.plot([0,0], [-0.25, 1], 'k--', lw=0.4)
        plt.xlim([0, 2.5])
        plt.ylim([-0.25, 1])
        plt.xlabel('Test EEG time / s')
        plt.ylabel('Pred EEG time / s')
        plt.title('Encoding accuracies')
        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{layer}_enc_acc'))     