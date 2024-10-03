import os
whiten = False
networks = 'Alexnet'
layer = 'all'
load_dir = f'output/sleemory_retrieval_vox/pred_eeg_ridge_PCA_whiten{whiten}/{networks}-{layer}/'
subs = range(2, 27)
for sub in subs:
    if sub != 17:
        old_fname = f'{networks}_pred_eeg_sub-{sub:03d}.mat'
        new_fname = f'{networks}-{layer}_pred_eeg_sub-{sub:03d}.mat'
        if os.path.exists(f'{load_dir}/{old_fname}'):
            os.rename(f'{load_dir}/{old_fname}', f'{load_dir}/{new_fname}')