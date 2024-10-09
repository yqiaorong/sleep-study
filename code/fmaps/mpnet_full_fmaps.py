"""This script extracts text features from filtered captions generaated by BLIP-2
by using model gpt-neo."""

import os
import argparse
import pandas as pd
import numpy as np
import scipy

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None, type=str)
args = parser.parse_args()

DNNetworks = 'mpnet'

print('')
print(f'>>> Sleemory images full feature maps {DNNetworks} <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')

# =============================================================================
# Load captions
# =============================================================================
    
capt_df = pd.read_csv(f'dataset/sleemory_{args.dataset}/BLIP-2_captions.csv', index_col=0)
print(capt_df)

# =============================================================================
# Load model
# =============================================================================

from superguse.get_embeddings import get_guse_data, get_mpnet_data
embeddings, stims, sentences = get_mpnet_data(f'dataset/sleemory_{args.dataset}/')
print(np.asarray(stims))
print(np.asarray(sentences))
print(np.asarray(embeddings).shape)

fmaps = {'imgs_all': stims, 'fmaps': embeddings}

# Save feature maps
save_dir = f'dataset/sleemory_{args.dataset}/dnn_feature_maps/full_feature_maps/{DNNetworks}'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
scipy.io.savemat(f'{save_dir}/{DNNetworks}_fmaps.mat', fmaps) 