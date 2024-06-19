import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse
from func import plot2D

parser = argparse.ArgumentParser()
parser.add_argument('--num_feat', default=1000, type=str)
parser.add_argument('--sub', default=3, type=int)
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



data = np.load(f'output/sleemory_retrieval/whiten_eeg_original/whiten_test_eeg_sub-{sub:03d}.npy', 
               allow_pickle=True).item()
eegs_sub = data['whitened_data']
imgs_sub = data['imgs_all']
del data

# Load predicted test eeg from the encoding model
pred_eeg_dir = f'output/sleemory_retrieval/test_pred_eeg/pred_eeg_with_{num_feat}feats.npy'
pred_eeg_all = np.load(pred_eeg_dir, allow_pickle=True).item() # dict of 8 layers
pred_eeg = pred_eeg_all[layer] # (4, ch, pred_time)
pred_time = pred_eeg.shape[2]

# List of imgs names
imgs_names = os.listdir('dataset/sleemory_retrieval/image_set')
imgs_names = [name[:-4] for name in imgs_names]


# PM and AM sessions
for ses in range(2):
    eegs_part = eegs_sub[ses] # (100, ch, test_time)
    imgs_part = np.squeeze(imgs_sub[:,ses][0]) # (100,)
    
    num_trial, test_time = eegs_part.shape[0], eegs_part.shape[2]
    enc_acc = np.empty((num_trial, pred_time, test_time))
    
    for trial_idx in tqdm(range(num_trial), desc='trials'):
        
        img_idx = np.where(imgs_names == imgs_part[trial_idx])
        
        # Correlations
        for t_test in range(test_time):
            test_val = pd.Series(eegs_part[trial_idx, :, t_test])
            for t_pred in range(pred_time):
                pred_val = pd.Series(np.squeeze(pred_eeg[img_idx, :, t_pred])) # (ch, pred_time)
                enc_acc[trial_idx, t_pred, t_test] = test_val.corr(pred_val)
                    
        # Save data
        save_dir = f'output/sleemory_retrieval/enc_acc/sub-{sub:03d}/ses-{ses}/{num_feat}feats_whiten{whiten}'
        if os.path.isdir(save_dir) == False:
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f'{layer}_enc_acc')
            
        np.save(save_path, enc_acc)
        
        # Average the results over stimuli
        avg_enc_acc = np.mean(enc_acc, axis=0)
        
        # Plot all 2D results 
        plot2D(avg_enc_acc, (10, 5), [-0.25, 1], [0, 2.5], 
              'Pred EEG time / s', 'Test EEG time / s', save_path)