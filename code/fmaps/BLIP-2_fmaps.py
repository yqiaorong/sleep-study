from PIL import Image
import os
import torch
from torchvision import transforms as trn
import argparse
from tqdm import tqdm
import scipy

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
# Create dataset
# =============================================================================
 
# Define the transform for images (Same as Alexnet)
size = 224
centre_crop = trn.Compose([
	trn.Resize((size,size)),
	trn.ToTensor(),
	# trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    
# =============================================================================
# Image feature maps
# =============================================================================

# Load model
from transformers import AutoProcessor, Blip2Model
model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b") # customize
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", # customize
                                          do_rescale=False) 

# Use GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

img_dir = f'dataset/sleemory_{args.dataset}/image_set/'

# Extract feature maps
all_feats = []
for img_name in tqdm(os.listdir(img_dir)):
    img = Image.open(os.path.join(img_dir, img_name)).convert('RGB')
    img = centre_crop(img) # transform
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        feats = outputs.last_hidden_state # convert output to tensor
        all_feats.append(feats)
    all_fmaps = torch.cat(all_feats).cpu().numpy() # change tensor to array
print(all_fmaps.shape) # (img, token, feat)

# Save feature maps
save_dir = f'dataset/sleemory_{args.dataset}/dnn_feature_maps'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
scipy.io.savemat(f'{save_dir}/{DNNetworks}_fmaps.mat', {'fmaps': all_fmaps}) 