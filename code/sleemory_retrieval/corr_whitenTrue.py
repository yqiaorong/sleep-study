"""
Correlations between the predicted retrieval session EEG and the 
real RE-ORDERED retrieval session EEG with whitening. The script processes one
subject one layer at a time. The each image result is saved in individual
subject in two sessions (AM & PM): sub >> ses >> whitenFalse >> image. 
"""

import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse
from func import plot2D

parser = argparse.ArgumentParser()
parser.add_argument('--num_feat', default=1000, type=str)
parser.add_argument('--sub', default=2, type=int)
parser.add_argument('--layer', default='conv5', type=str)
args = parser.parse_args()

print('')
print(f'>>> Correlation of sleemory retrieval (whiten True) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')



whiten = True
sub = args.sub
num_feat = args.num_feat
layer = args.layer



data = np.load(f'output/sleemory_retrieval/whiten_eeg/whiten_test_eeg_sub-{sub:03d}.npy', allow_pickle=True).item()
eegs = data['whitened_data']
imgs_names = data['imgs_all']
del data


# Load predicted test eeg from the encoding model
pred_eeg_dir = f'output/sleemory_retrieval/test_pred_eeg/pred_eeg_with_{num_feat}feats.npy'
pred_eeg_all = np.load(pred_eeg_dir, allow_pickle=True).item() # dict of 8 layers
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
        save_path = os.path.join(save_dir, f'{name}_{layer}_enc_acc')
            
        np.save(save_path, enc_acc)
        
        # Average the results over stimuli
        avg_enc_acc = np.mean(enc_acc, axis=0)
        
        # Plot all 2D results 
        plot2D(avg_enc_acc, (10, 5), [-0.25, 1], [0, 2.5], 
              'Pred EEG time / s', 'Test EEG time / s', save_path)