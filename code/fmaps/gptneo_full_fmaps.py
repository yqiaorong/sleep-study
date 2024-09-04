"""This script extracts text features from filtered captions generaated by BLIP-2
by using model gpt-neo."""

import os
import torch
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import scipy

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None, type=str)
args = parser.parse_args()

DNNetworks = 'gptneo'

print('')
print(f'>>> Sleemory images full feature maps {DNNetworks} <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')

# =============================================================================
# Load captions
# =============================================================================
    
capt_df = pd.read_csv(f'dataset/sleemory_{args.dataset}/BLIP-2_captions.csv')

# Re-order captions according to the image order
img_names = os.listdir(f'dataset/sleemory_{args.dataset}/image_set')
capts = []
for name in img_names:
    capts.append(capt_df.loc[capt_df['img_names'] == name]['gen_texts'].iloc[0])

# =============================================================================
# Getting feature maps and captions
# =============================================================================

# Load model
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B",
                                             torch_dtype=torch.float32)
model.eval() 
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

# =============================================================================
# Extract capt features
# =============================================================================

# Initialize the dict of fmaps
fmaps = {}  
fmaps['layer_0_embeddings'] = []
for idx in range(model.config.num_layers):
    attention_type = model.config.attention_layers[idx]
    fmaps[f'layer_{idx+1}_{attention_type}'] = []

# Iterate over captions
for icapt, capt in enumerate(tqdm(capts)):                       
    input = tokenizer(capt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**input, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        for ilayer, layer in enumerate(hidden_states):
            if ilayer != 0:
                attention_type = model.config.attention_layers[ilayer-1]
            feat = torch.mean(layer, axis=1).numpy() # take the avg across capt strings
            
            if icapt == 0:
                if ilayer == 0:
                    fmaps[f'layer_{ilayer}_embeddings'] = feat
                else:
                    fmaps[f'layer_{ilayer}_{attention_type}'] = feat
            else:
                if ilayer == 0:
                    fmaps[f'layer_{ilayer}_embeddings'] = np.concatenate((fmaps[f'layer_{ilayer}_embeddings'], feat), axis=0)
                else:
                    fmaps[f'layer_{ilayer}_{attention_type}'] = np.concatenate((fmaps[f'layer_{ilayer}_{attention_type}'], feat), axis=0) 

fmaps['imgs_all'] = img_names
fmaps['captions'] = capts

# Save feature maps
save_dir = f'dataset/sleemory_{args.dataset}/dnn_feature_maps/full_feature_maps/GPTNeo'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
scipy.io.savemat(f'{save_dir}/{DNNetworks}_fmaps.mat', fmaps) 