import os
networks = 'ResNet'
layers = ['maxpool', 'layer3']
# layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8','all']

subs = range(2, 27)

for layer in layers:
    for sub in subs:
        if sub != 17:
            os.system(f'python3 code/validation/model_ridge_fracs_PCA.py --sub {sub} --networks {networks} --layer_name {layer}')
    os.system(f'python3 code/validation/avg_corr_each_model.py --networks {networks}-{layer}')

# for sub in subs:
#     if sub != 17:
#         os.system(f'python3 code/validation/{networks}_ridge_fracs_PCA.py --sub {sub}')
# os.system(f'python3 code/validation/avg_corr_each_model.py --networks {networks}')