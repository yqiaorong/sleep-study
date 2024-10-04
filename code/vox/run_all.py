import os

networks = 'ResNet' # [ ResNet / Alexnet / mpnet / GPTNeo ]
layers = ['maxpool','layer3']
# layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8','all']

subs = range(2, 27)

for sub in subs:
    if sub != 17:
        for layer in layers:
            os.system(f'python3 code/vox/model_ridge_fracs_PCA.py --sub {sub} --networks {networks} --layer_name {layer}')

# for sub in subs:
#     if sub != 17:
#         os.system(f'python3 code/vox/{networks}_ridge_fracs_PCA.py --sub {sub}')