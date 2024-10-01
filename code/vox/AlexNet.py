import os

layers = [# 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8', 
          'all']

subs = range(2, 27)
for layer in layers:
    # os.system(f'python3 code/fmaps/Alexnet_best_fmaps_PCA.py --layer {layer}')
    for sub in subs:
        if sub != 17:
            os.system(f'python3 code/vox/AlexNet_ridge_fracs_PCA.py --sub {sub} --layer {layer}')