import numpy as np
import scipy
from matplotlib import pyplot as plt

# sleemory_retrieval
data = np.load(f'output/sleemory_retrieval/test_pred_eeg/pred_eeg_with_1000feats.npy', 
               allow_pickle=True).item()
for key, value in data.items():
    print(key, value.shape)
print('')

# sleemory_localiser
layers = ['conv1', 'conv2', 
          #'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'
          ]
eeg_localiser = []
for layer in layers:
    # .mat file:
    # data = scipy.io.loadmat(f'output/sleemory_localiser/test_pred_eeg/1000feats/{layer}_eeg.mat')
    # .npy file:
    data = np.load(f'output/sleemory_localiser/test_pred_eeg/1000feats/{layer}_eeg.npy', 
               allow_pickle=True).item()
    
    for key, value in data.items():
        if key == 'pred_eeg':
            print(layer, key, data[key].shape)
            eeg_localiser.append(data['pred_eeg'])
print(np.array_equal(eeg_localiser[0], eeg_localiser[1])) 
    
    
    
# Check encoding accuracy 
enc_data = []
selected_layers = ['conv5', 'conv2']
for l in selected_layers:
    data = np.load(f'output/sleemory_localiser/enc_acc/1000feats/{l}_enc_acc.npy', 
               allow_pickle=True)
    enc_data.append(np.mean(data, axis=0))
    print(data.shape)
print(np.array_equal(enc_data[0], enc_data[1]))

# Plot the difference in encoding accuracy
enc_diff = enc_data[0]-enc_data[1]
fig = plt.figure(figsize=(6, 5))
im = plt.imshow(enc_diff, cmap='viridis',
				extent=[-0.25, 1, -0.25, 1], 
                origin='lower', aspect='auto')
cbar = plt.colorbar(im)
cbar.set_label('Values')

# Plot borders
plt.plot([-0.25, 1], [0,0], 'k--', lw=0.4)
plt.plot([0,0], [-0.25, 1], 'k--', lw=0.4)
plt.xlim([-0.25, 1])
plt.ylim([-0.25, 1])
plt.xlabel('Test EEG time / s')
plt.ylabel('Pred EEG time / s')
plt.title(f'Encoding accuracy difference between layers {selected_layers[0]} and {selected_layers[1]}')
fig.tight_layout()
plt.show()
