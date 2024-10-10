import os

# network = 'ResNet' 
# layers = ['maxpool','layer3']

# network = 'Alexnet'
# layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8','all']

networks = ['BLIP-2', 'CLIP', # 'mpnet', 'GPTNeo'
            ]



subs = range(2, 27)

# for sub in subs:
#     if sub != 17:
#         for layer in layers:
#             if not os.path.exists(f'output/sleemory_retrieval_vox/pred_eeg_ridge_PCA_whitenFalse/{network}-{layer}/{network}-{layer}_pred_eeg_sub-{sub:03d}.mat'):
#                 os.system(f'python3 code/validation/pred_retrieval_eeg.py --sub {sub} --networks {network} --layer_name {layer}')

for sub in subs:
    if sub != 17:
        for network in networks:
            if not os.path.exists(f'output/sleemory_retrieval_vox/pred_eeg_ridge_PCA_whitenFalse/{network}/{network}_pred_eeg_sub-{sub:03d}.mat'):
                os.system(f'python3 code/validation/pred_retrieval_eeg.py  --sub {sub} --networks {network}')