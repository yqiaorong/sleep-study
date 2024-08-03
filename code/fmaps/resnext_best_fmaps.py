"""This script runs resnext_best_fmaps_chunk.py, which extracts the best
feature maps of individual layers."""

import scipy
import os
import argparse

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--num_feat', default=3000, type=int)
parser.add_argument('--whiten',   default=True, type=bool)
args = parser.parse_args()

print('')
print(f'>>> Feature selection of sleemory images feature maps (resnext) for all layers <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')

# Load layer names
sample_fmaps = scipy.io.loadmat('dataset/sleemory_localiser/dnn_feature_maps/full_feature_maps/ResNet/img_0.mat')
layers = list(sample_fmaps.keys())
num_layers = len(layers[3:])
print(f'The number of layers: {num_layers}')
del sample_fmaps, layers

# Run the script
layer_start_indices = range(0, num_layers, 50)

for idx in layer_start_indices:
    os.system(f'python code/fmaps/resnext_best_fmaps_chunk.py'+
              f' --whiten {args.whiten} --layer_start_idx {idx} --num_feat {args.num_feat}')