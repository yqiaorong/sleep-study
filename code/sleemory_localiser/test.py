import numpy as np

# sleemory_retrieval
data = np.load(f'output/sleemory_retrieval/test_pred_eeg/pred_eeg_with_1000feats.npy', 
               allow_pickle=True).item()
for key, value in data.items():
    print(key, value.shape)
print('')

# sleemory_localiser
layer = 'conv5'
data = np.load(f'output/sleemory_localiser/test_pred_eeg/{layer}_eeg.npy', 
               allow_pickle=True).item()
for key, value in data.items():
    print(layer, key, value.shape)