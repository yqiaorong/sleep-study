import torch
from PIL import Image
from torchvision import transforms as trn
import os
from tqdm import tqdm
import argparse
import scipy

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None, type=str)
parser.add_argument('--img_name', default=None, type=str)
args = parser.parse_args()

DNNetworks = 'ResNet'

print('')
print(f'>>> Extract sleemory images feature maps ({DNNetworks}) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
 
# =============================================================================
# Define image preprocessing
# =============================================================================

size = 224
img_prepr = trn.Compose([
	trn.Resize((size,size)),
	trn.ToTensor(),
	trn.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =============================================================================
# Load the networks
# =============================================================================

model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')

# Use GPU otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print(device)

# =============================================================================
# Extract features
# =============================================================================

# Set up the features
all_feats = {}
def hook_fn(module, input, output):
    layer_name = f"{module.__class__.__name__}_{id(module)}"
    print(layer_name)
    all_feats[layer_name] = output
    
def register_hooks(model):
    for _, module in model.named_modules():
        module.register_forward_hook(hook_fn)
        
register_hooks(model)        

# Select img
img_dir = f'/home/simon/Documents/gitrepos/shannon_encodingmodelsEEG/dataset//sleemory_{args.dataset}/image_set/'
img_name = args.img_name
print(img_name)

# Extract
img = Image.open(os.path.join(img_dir, img_name)).convert('RGB')
img = img_prepr(img) # transform
input = img.unsqueeze(0).to(device) # create a mini-batch as expected by the model
with torch.no_grad():
    _ = model(input)
        
# Reshape the features
for layer_name, feat_vals in all_feats.items():
    if len(feat_vals.shape) == 4:
        all_feats[layer_name] = feat_vals.flatten(start_dim=1)  

# Convert to numpy array
final_fmaps = {}
for i, (layer_name, feat_vals) in enumerate(all_feats.items()):  
    feat_vals = feat_vals.cpu()
    final_fmaps[layer_name.split('_')[0]+f'_{i}'] = feat_vals.numpy()
del all_feats

# Save feature maps
save_dir = f'dataset/sleemory_{args.dataset}/dnn_feature_maps/full_feature_maps/{DNNetworks}'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
saved_fname = img_name.split('.')[0]
print(saved_fname)
scipy.io.savemat(f'{save_dir}/{saved_fname}_feats.mat', final_fmaps)