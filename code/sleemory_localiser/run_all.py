import os

for i in range(1,6):
    os.system(f'python code/sleemory_localiser/enc_corr.py --layer_name conv{i}')
for i in range(6,9):
    os.system(f'python code/sleemory_localiser/enc_corr.py --layer_name fc{i}')