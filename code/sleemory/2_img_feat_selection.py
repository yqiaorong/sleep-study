import os
import pickle
import argparse
import numpy as np
from func import load_sleemory_full_fmaps

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--layer_name', default='conv5', type=str)
parser.add_argument('--num_feat', default=300, type=int)
args = parser.parse_args()

print('')
print('Feature selection of sleemory images feature maps <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# =============================================================================
# Load the training feature maps
# =============================================================================

fmaps = load_sleemory_full_fmaps(args)

# Fit feature selection model and transform
if args.num_feat == -1:
    best_feat = fmaps[args.layer_name]
else:
    feat_model_dir = os.path.join('dataset', 'THINGS_EEG2', 'model', 
                                  f'feat_model_{args.num_feat}_sleemory.pkl')
    with open(feat_model_dir, 'rb') as f:
        feature_selection = pickle.load(f)
        best_feat = feature_selection.transform(fmaps[args.layer_name])

print(f'The new training fmaps shape {best_feat.shape}')
del fmaps

# =============================================================================
# Save new features
# =============================================================================

save_dir = os.path.join('dataset','temp_sleemory')

# Save new features
np.save(os.path.join(save_dir, 'dnn_feature_maps', 
                     f'new_feature_maps_{args.num_feat}'), best_feat) 