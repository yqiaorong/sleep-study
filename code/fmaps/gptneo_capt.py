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
print(f'>>> Sleemory images feature maps and captions {DNNetworks} <<<')
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

fmaps = None
for capt in tqdm(capts):                       
    input = tokenizer(capt, return_tensors="pt")
    with torch.no_grad():
         outputs = model(**input, output_hidden_states=True)
         hidden_states = outputs.hidden_states
         # Choose the last layer and take the avg across capt strings
         feat = np.squeeze(torch.mean(hidden_states[-1], axis=1))
         if fmaps == None:
             fmaps = feat
         else:
             fmaps = torch.vstack((fmaps, feat)) # (num_img, num_feat)
print(fmaps.shape)

# Save feature maps
save_dir = f'dataset/sleemory_{args.dataset}/dnn_feature_maps'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
scipy.io.savemat(f'{save_dir}/{DNNetworks}_fmaps.mat', {'fmaps': fmaps}) 