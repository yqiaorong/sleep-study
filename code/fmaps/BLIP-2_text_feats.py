"""This script extracts the text features using BLIP-2 model. There are two options:
using the image names as text prompts, or MANUALLY MODIFIED generated texts using
BLIP-2 model as text prompts.

!!! The problem of the features extracted here is that they depend on the length of 
generated text which makes it difficult to train the encoding model.!!!

In summary, there are three types of text features:
    from automatically generated texts, (use BLIP-2_capt.py)
    from manually modified generated texts,
    from image names,
"""

import os
import torch
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',   default=None, type=str)
parser.add_argument('--text_type', default=None, type=str) # [imgname, captions]
args = parser.parse_args()

DNNetworks = 'BLIP-2'

print('')
print(f'>>> Extract sleemory images feature maps {DNNetworks} <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# =============================================================================
# Load captions
# =============================================================================

if args.text_type == 'imgname':
    capts = os.listdir(f'dataset/sleemory_{args.dataset}/image_set/')
    capts = [capt[:-4] for capt in capts]
    
elif args.text_type == 'captions':
    capt_df = pd.read_csv(f'dataset/sleemory_{args.dataset}/{DNNetworks}_captions.csv')
    capts = capt_df['gen_texts']

# =============================================================================
# Load cmodel
# =============================================================================

from transformers import Blip2Model, AutoTokenizer

model_text = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")     
tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")

# Use GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model_text.to(device)

# =============================================================================
# Extract feature maps
# =============================================================================

img_feats, text_feats = [], []
for capt in tqdm(capts):
    ## Input
    text_inputs = tokenizer([capt], padding=True, return_tensors="pt")
    
    ## Output
    with torch.no_grad():
        # text outputs is dict
        text_outputs = model_text.get_text_features(**text_inputs, 
                                               return_dict=True, output_hidden_states=True)

        # logits = text_outputs['logits'] #array shape (img, num_words, feats)
        # past_key_values = text_outputs['past_key_values'] # tuple 'I don't think it's useful
        hidden_states = np.asarray(text_outputs['hidden_states']).squeeze(1) # tuple --> array (layers, num_words, feats)
        print(hidden_states.shape)
        text_feats.append(hidden_states)

# =============================================================================
# Save feature maps
# =============================================================================

save_dir = f'dataset/sleemory_{args.dataset}/dnn_feature_maps'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
scipy.io.savemat(f'{save_dir}/{DNNetworks}_text_fmaps_{args.text_type}.mat', 
                 {'lang_model_feats': text_feats})