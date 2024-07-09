import os
import torch
import argparse
from tqdm import tqdm
import numpy as np

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None, type=str)
args = parser.parse_args()

DNNetworks = 'BLIP-2'

print('')
print(f'>>> Extract sleemory images feature maps {DNNetworks} <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


  
# =============================================================================
# Image feature maps
# =============================================================================

# Load model
from transformers import Blip2Model, AutoTokenizer

model_text = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")      # customize
tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")

# Use GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model_text.to(device)

img_dir = f'dataset/sleemory_{args.dataset}/image_set/'

# Extract feature maps
img_feats, text_feats = [], []
for text in tqdm(os.listdir(img_dir)):
    ## Input
    text_inputs = tokenizer([text[:-4]], padding=True, return_tensors="pt")
    
    ## Output
    with torch.no_grad():
        # text outputs is dict
        text_outputs = model_text.get_text_features(**text_inputs, 
                                               return_dict=True, output_hidden_states=True)

        logits = text_outputs['logits'] #array shape (img, ?, feats)
        print(logits.shape)
        # past_key_values = text_outputs['past_key_values'] # tuple 'I don't think it's useful
        hidden_states = np.asarray(text_outputs['hidden_states']).squeeze(1) # tuple --> array (layers, ?, feats)
        print(hidden_states.shape)
        text_feats.append(hidden_states)

# Save feature maps
save_dir = f'dataset/sleemory_{args.dataset}/dnn_feature_maps'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.savez(f'{save_dir}/{DNNetworks}_text_fmaps.mat', *text_feats) 