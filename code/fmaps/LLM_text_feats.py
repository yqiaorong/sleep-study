"""This script extracts the text features using BLIP-2 / CLIP model. 

In summary, there are three types of text features:
    from automatically generated texts, (use BLIP-2_capt.py)
    from manually modified generated texts,
    from image names,
"""

import os
import torch
import torch.nn as nn
import argparse
import numpy as np
import pandas as pd
import scipy

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--networks', default=None, type=str) # [BLIP-2 / CLIP]
parser.add_argument('--dataset',  default=None, type=str)
args = parser.parse_args()

print('')
print(f'>>> Extract sleemory images feature maps {args.networks} <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# =============================================================================
# Load captions
# =============================================================================

capt_df = pd.read_csv(f'dataset/sleemory_{args.dataset}/BLIP-2_captions.csv')
imgs_all, capts = capt_df['img_names'], capt_df['gen_texts']

# =============================================================================
# Load cmodel
# =============================================================================

from transformers import AutoTokenizer

if args.networks == 'BLIP-2':
    from transformers import Blip2Model
    model_text = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")     
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
    print('BLIP-2 model ready!')
elif args.networks == 'CLIP':
    from transformers import CLIPModel
    model_text = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    print('CLIP model ready!')
    
# for name, layer in model_text.named_modules():
#     if hasattr(layer, 'weight') and layer.weight is not None:
#         print(f"Layer name: {name}")
#         print(f"Layer weight shape: {layer.weight.size()}")
#     # print('')
#     # if isinstance(layer, torch.nn.Conv2d):
#     #     print(f"Layer name: {name}")
#     #     print(f"Cardinality (Groups): {layer.groups}")
#     print('-' * 50)

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1)  # learnable attention weights

    def forward(self, hidden_states):
        # hidden_states: (batch_size, Nwords, hidden_dim)
        attn_scores = self.attention_weights(hidden_states)  # (batch_size, Nwords, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (batch_size, Nwords, 1)
        # weighted sum of hidden states
        weighted_hidden_states = hidden_states * attn_weights  # element-wise multiplication
        pooled_output = torch.sum(weighted_hidden_states, dim=1)  # (batch_size, hidden_dim)
        return pooled_output

atten_pool = AttentionPooling(hidden_dim=2560)

# Use GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'
model_text.to(device)

# =============================================================================
# Extract feature maps
# =============================================================================

flabels, text_feats = [], []
for icapt, capt in enumerate(capts):
    print(imgs_all[icapt], capt)
    flabels.append(imgs_all[icapt])
    ## Input
    text_inputs = tokenizer([capt], padding=True, return_tensors="pt")
    
    ## Output
    with torch.no_grad():
        # text outputs is dict
        text_outputs = model_text.get_text_features(**text_inputs, return_dict=True, output_hidden_states=True)

        if args.networks == 'BLIP-2':
            # logits          = text_outputs['logits']                              
            # past_key_values = text_outputs['past_key_values']                      
            # hidden_states   = text_outputs['hidden_states'])   # tuple, (batch size, sequential length, feats)

            hidden_states = torch.stack(text_outputs['hidden_states'], dim=0).squeeze(1) # tensor 
            # Attention pooling
            pooled_hidden_states = atten_pool(hidden_states) # tensor
            del hidden_states
        elif args.networks == 'CLIP':
            pooled_hidden_states = text_outputs # tensor 
        
        text_feats.append(pooled_hidden_states)
text_feats = np.array(text_feats).reshape(len(flabels), -1) # (imgs, feats)
print(text_feats.shape)
del imgs_all, capts

# =============================================================================
# Save feature maps
# =============================================================================

save_dir = f'dataset/sleemory_{args.dataset}/dnn_feature_maps/full_feature_maps/{args.networks}/'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
scipy.io.savemat(f'{save_dir}/{args.networks}_fmaps.mat', {'fmaps': text_feats,
                                                           'imgs_all': flabels}) 