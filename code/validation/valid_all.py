import os
networks = ['BLIP-2', 'CLIP']

# layers = ['layer3', 'maxpool']
# layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8','all']

subs = range(2, 27)

# for sub in subs:
#     if sub != 17:
#         for layer in layers:
#             os.system(f'python3 code/validation/model_ridge_fracs_PCA.py --sub {sub} --networks {networks} --layer_name {layer}')
# os.system(f'python3 code/validation/avg_corr_each_model.py --networks {networks}-{layer}')

for sub in subs:
    if sub != 17:
        for network in networks:
            os.system(f'python3 code/validation/model_ridge_fracs_PCA.py --network {network} --sub {sub}')
for network in networks:
    os.system(f'python3 code/validation/avg_corr_each_model.py --networks {network}')