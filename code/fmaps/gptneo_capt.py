from PIL import Image
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

DNNetworks = 'BLIP-2'

print('')
print(f'>>> Sleemory images feature maps and captions {DNNetworks} <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')
    
# =============================================================================
# Getting feature maps and captions
# =============================================================================

# Load model
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

model_cap = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B",
                                              torch_dtype=torch.float32)
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")