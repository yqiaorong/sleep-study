import os

subs = range(2, 27)

networks = ['BLIP-2', 'CLIP', 'mpnet', 'gptneo']
for sub in subs:
    if sub != 17:
        for network in networks:
            if not os.path.exists(f'output/sleemory_retrieval_vox/pred_eeg_ridge_PCA_whitenFalse/{network}/{network}_pred_eeg_sub-{sub:03d}.mat'):
                os.system(f'python3 code/validation/pred_retrieval_eeg.py  --sub {sub} --networks {network}')

networks = ['ResNeXt' , 'AlexNet']
layers = [['maxpool','layer3', 'fc'], ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8','all']]
for i in range(2):
    for sub in subs:
        if sub != 17:
            for layer in layers[i]:
                if not os.path.exists(f'output/sleemory_retrieval_vox/pred_eeg_ridge_PCA_whitenFalse/{network[i]}-{layer}/{network[i]}-{layer}_pred_eeg_sub-{sub:03d}.mat'):
                    os.system(f'python3 code/validation/pred_retrieval_eeg.py --sub {sub} --networks {network[i]} --layer_name {layer}')