"""
Correlations between the predicted retrieval session EEG and the 
real RE-ORDERED retrieval session EEG without whitening. The script processes one
subject one layer at a time. The each image result is saved in individual
subject in two sessions (AM & PM): sub >> ses >> whitenFalse >> image. 
"""

import scipy.io
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_feat', default=1000, type=str)
parser.add_argument('--sub', default=2, type=int)
parser.add_argument('--layer', default='conv5', type=str)
args = parser.parse_args()

print('')
print(f'>>> Correlation of sleemory retrieval (whiten False) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')



whiten = False
sub = args.sub
num_feat = args.num_feat
layer = args.layer



load_path = f'dataset/sleemory_retrieval/preprocessed_data/sleemory_retrieval_dataset_sub-{sub:03d}'
data = scipy.io.loadmat(load_path)
eegs = data['ERP_all'] # (1, 2)
imgs = data['imgs_all'] # (1, 2)
del data

    
# List of imgs names
imgs_names = os.listdir('dataset/sleemory_retrieval/image_set')
imgs_names = [name[:-4] for name in imgs_names]


# Load predicted test eeg from the encoding model
pred_eeg_dir = f'output/sleemory_retrieval/test_pred_eeg/pred_eeg_with_{num_feat}feats.npy'
pred_eeg_all = np.load(pred_eeg_dir, allow_pickle=True).item() # dict of 8 layers
pred_eeg = pred_eeg_all[layer]
pred_time = pred_eeg.shape[2]


# PM and AM sessions
for ses in range(2):
    eegs_part = eegs[:, ses][0]
    imgs_part = imgs[:, ses][0][:,0]
    
    # Classify eeg data according to imgs names
    for i, name in enumerate(imgs_names):
        mask = imgs_part == name
        eeg = eegs_part[mask]
        
        num_trial, test_time = eeg.shape[0], eeg.shape[2]
        
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
            
        np.save(os.path.join(save_dir, f'{name}_{layer}_enc_acc'), enc_acc)
        
        # Average the results over stimuli
        avg_enc_acc = np.mean(enc_acc, axis=0)
        
        # Plot all 2D results of method 1
        fig = plt.figure(figsize=(10, 5))
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
        plt.savefig(os.path.join(save_dir, f'{name}_{layer}_enc_acc'))     