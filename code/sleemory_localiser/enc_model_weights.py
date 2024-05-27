import pickle
import os
import numpy as np
import scipy

# Weights dictionary
weights = {}

reg_dir = f'dataset/sleemory_localiser/model/reg_model'
layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
for l in layer_names:
    reg = pickle.load(open(os.path.join(reg_dir, f'{l}_reg_model.pkl'), 'rb'))
    w = reg.coef_
    # reshape weights
    w = np.reshape(w, (58, 363, 1000)) # [channel, time, feature]
    weights[l] = w
    
# Save weights
scipy.io.savemat(os.path.join(reg_dir, 'reg_weights.mat'), weights) 

# check
for key, values in weights.items():
    print(key, values.shape)