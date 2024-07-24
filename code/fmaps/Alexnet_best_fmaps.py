import os
import pickle
import argparse
import numpy as np
import scipy
from func import load_sleemory_full_fmaps
from sklearn.feature_selection import SelectKBest, f_regression

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--dataset',    default=None, type=str)
parser.add_argument('--num_feat',   default=None, type=int)
parser.add_argument('--whiten',     default=False,type=bool)
args = parser.parse_args()

print('')
print('>>> Feature selection of sleemory images feature maps <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')

# Set up
if args.whiten == False:
    whiten = ''
else:
    whiten = 'whiten_'
    
# =============================================================================
# Load the feature maps
# =============================================================================

fmaps = load_sleemory_full_fmaps(args)

# =============================================================================
# Feature selection
# =============================================================================

# save dir
save_dir = f'dataset/{args.dataset}/dnn_feature_maps'
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)
         
         
# Fit feature selection model and transform
all_layers = fmaps.keys()
for layer in all_layers:
    print(layer)
    if args.num_feat == -1:
        best_feat = fmaps[layer]
    else:
        model_fname = f'{layer}-best-{args.num_feat}_{whiten}feat_model.pkl'
        # layers names
        if args.dataset == 'sleemory_localiser':
            # Load eeg data
            data_fname = f'unique_{whiten}eeg.mat'
            print(data_fname)
            data = scipy.io.loadmat(f'output/{args.dataset}/whiten_eeg/{data_fname}')
            eeg_data = data[f'{whiten}eeg']
            # Average across time to shape (img, ch,)
            eeg_data = np.mean(eeg_data, -1)
            # Average across channel to shape (img,)
            eeg_data = np.mean(eeg_data, -1)
            print('EEG data shape (img,)', eeg_data.shape)
            
            # Build the feature selection model
            feature_selection = SelectKBest(f_regression, 
                                            k=args.num_feat).fit(fmaps[layer], eeg_data)
            best_feat = feature_selection.transform(fmaps[layer])
            
            # Save the model
            model_dir = f'dataset/{args.dataset}/model/best_fmaps_model'
            if os.path.isdir(model_dir) == False:
                os.makedirs(model_dir)
            print(model_fname)
            pickle.dump(feature_selection, open(f'{model_dir}/{model_fname}', 'wb'))
            del feature_selection
            
        elif args.dataset == 'sleemory_retrieval':
            # Apply the model
            print(model_fname)
            with open(f'dataset/sleemory_localiser/model/best_fmaps_model/{model_fname}', 'rb') as f:
                feature_selection = pickle.load(f)
                best_feat = feature_selection.transform(fmaps[layer])
                
        print(f'The new fmaps of layer {layer} has shape {best_feat.shape}')
        
        # Save new features
        best_fmaps_fname = f'{layer}-best-{args.num_feat}_{whiten}fmaps.mat'
        print(best_fmaps_fname)
        scipy.io.savemat(f'{save_dir}/{best_fmaps_fname}', {'fmaps': best_feat}) 
    print('')