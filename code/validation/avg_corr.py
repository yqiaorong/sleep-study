import os

networks = ['Alexnet-conv1', 'Alexnet-conv2','Alexnet-conv3','Alexnet-conv4','Alexnet-conv5',
            'Alexnet-fc5','Alexnet-fc6','Alexnet-fc7','Alexnet-all',
            'GPTNeo', 'ResNet-fc']

for network in networks:
     os.system(f'python3 code/validation/avg_corr_each_model.py --networks {network}')