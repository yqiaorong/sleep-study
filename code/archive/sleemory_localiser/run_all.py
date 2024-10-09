"""This script computes the Pearson coeffs between test eeg and pred eeg of all 8 
layers in Alexnet. The script is archived. Use code_matlab to speed up the calculations.
"""

import os

for i in range(1,6):
    os.system(f'python code/sleemory_localiser/enc_corr.py --layer_name conv{i}')
for i in range(6,9):
    os.system(f'python code/sleemory_localiser/enc_corr.py --layer_name fc{i}')