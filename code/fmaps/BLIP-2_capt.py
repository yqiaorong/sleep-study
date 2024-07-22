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
from transformers import Blip2Processor, Blip2ForConditionalGeneration , AutoTokenizer

model_cap = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", 
            # load_in_8bit=True, device_map={"": 0}, # Require GPU
            torch_dtype=torch.float32) 
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")

# Use CPU or GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model_cap.to(device)

img_dir = f'dataset/sleemory_{args.dataset}/image_set/'

# Extract feature maps and captions
gen_texts, lang_feats = [], []
for img_name in tqdm(os.listdir(img_dir)):
    # Input
    img = Image.open(os.path.join(img_dir, img_name)).convert('RGB')
    img_input = processor(images=img, return_tensors="pt").to(device, torch.float16)
    
    # Generate caption from image
    gen_ids = model_cap.generate(**img_input)
    gen_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    print(gen_text)
    gen_texts.append(gen_text)

    # Get lang model output
    input_ids = tokenizer(img_name[:-4], padding=True, return_tensors="pt")
    output = model_cap(img_input['pixel_values'], input_ids['input_ids'])
    lang_model_out = output.language_model_outputs
    lang_feat = lang_model_out.logits
    lang_feats.append(lang_feat.squeeze().detach().numpy())
lang_feats = np.asarray(lang_feats)

# =============================================================================
# Save
# =============================================================================

save_dir = f'dataset/sleemory_{args.dataset}'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

# Save the captions
df = pd.DataFrame({'img_names': os.listdir(img_dir), 'gen_texts': gen_texts})
df.to_csv(f'{save_dir}/{DNNetworks}_captions.csv')

# Save the lang feats
np.savez(f'{save_dir}/{DNNetworks}_lang_model_feats.npz', *lang_feats)
# scipy.io.savemat(f'{save_dir}/{DNNetworks}_fmaps.mat', {'lang_model_feats': lang_feats}) 