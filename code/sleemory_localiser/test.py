import numpy as np
layer = 'conv5'

data = np.load(f'output/sleemory_localiser/test_pred_eeg/{layer}_eeg.npy', 
               allow_pickle=True).item()
print(data['train_eeg'].shape, data['test_eeg'].shape, data['pred_eeg'].shape)

data = np.load(f'output/sleemory_retrieval/test_pred_eeg/{layer}_eeg.npy', 
               allow_pickle=True).item()
print(data['pred_eeg'].shape)