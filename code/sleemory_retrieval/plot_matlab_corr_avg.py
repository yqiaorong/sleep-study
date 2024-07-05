import scipy.io
from func import plot2D
import numpy as np
import os
from tqdm import tqdm


# whiten
data_type = '' # [/_whitenFalse]

save_dir = f'output/sleemory_retrieval/enc_acc_avg{data_type}'
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)



layers = [# 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8', 
          'CLIP']
for layer in layers:
    
    # Get the avg data
    sub_avg_data = []
    for sub in tqdm(range(2,27), desc='subject'):
        if sub != 17:
            data = scipy.io.loadmat(f'output/sleemory_retrieval/enc_acc/sub-{sub:03d}/{layer}_enc_acc_all{data_type}.mat')
            sub_avg_data.append(np.mean(data['enc_acc'], axis=1)) # (2, test_time, pred_time)
    sub_avg_data = np.mean(sub_avg_data, axis=0)
    
    # Separate plot
    for idx, ses in enumerate(['PM', 'AM']):
        save_path = os.path.join(save_dir, f'{ses}ses_{layer}_enc_acc')
        plot2D(sub_avg_data[idx].T, (10, 5), [-0.25, 1], [0, 2.5], 
                'Pred EEG time / s', 'Test EEG time / s', save_path)
        
    # Plot the difference
    differ_path = os.path.join(save_dir, f'differ_{layer}_enc_acc')
    differ = sub_avg_data[1]-sub_avg_data[0] # AM-PM
    plot2D(differ.T, (10, 5), [-0.25, 1], [0, 2.5], 
                'Pred EEG time / s', 'Test EEG time / s', differ_path)