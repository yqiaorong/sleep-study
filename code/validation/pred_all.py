import os

# network = 'ResNet' 
# layers = ['maxpool','layer3']

# network = 'Alexnet'
# layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8','all']

networks = ['BLIP-2', 'CLIP', 'mpnet', 'GPTNeo']



subs = range(2, 27)

# for sub in subs:
#     if sub != 17:
#         for layer in layers:
#             os.system(f'python3 code/validation/pred_retrieval_eeg.py --sub {sub} --networks {network} --layer_name {layer}')

for sub in subs:
    if sub != 17:
        for network in networks:
            os.system(f'python3 code/validation/pred_retrieval_eeg.py  --sub {sub} --networks {network}')