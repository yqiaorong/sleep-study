import scipy.io
from func import plot2D
import numpy as np
import os
from tqdm import tqdm

layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
for sub in tqdm(range(18, 27), desc='subject'):
    if sub != 17:
        for layer in layers:
            data = scipy.io.loadmat(f'output/sleemory_retrieval/enc_acc/sub-{sub:03d}/{layer}_enc_acc_all.mat')
            avg_data = np.mean(data['enc_acc'], axis=1) # (2, test_time, pred_time)

            save_dir = f'output/sleemory_retrieval/enc_acc/sub-{sub:03d}/plot'
            if os.path.isdir(save_dir) == False:
                os.makedirs(save_dir)
            for idx, ses in enumerate(['PM', 'AM']):
                save_path = os.path.join(save_dir, f'{ses}ses_{layer}_enc_acc')
                plot2D(avg_data[idx].T, (10, 5), [-0.25, 1], [0, 2.5], 
                        'Pred EEG time / s', 'Test EEG time / s', save_path)