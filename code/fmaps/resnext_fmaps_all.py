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
model.eval()

# Use GPU otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# =============================================================================
# Extract features
# =============================================================================

# Set up the features
all_feats = {}
def hook_fn(module, input, output):
    layer_name = f"{module.__class__.__name__}_{id(module)}"
    # print(layer_name)
    all_feats[layer_name] = output
    
def register_hooks(model):
    for _, module in model.named_modules():
        module.register_forward_hook(hook_fn)
        
register_hooks(model)        
# print(model.named_modules())

# Extract
img_dir = f'dataset/sleemory_{args.dataset}/image_set/'
fmaps = None
for i, img_name in tqdm(enumerate([os.listdir(img_dir)[0]])):
    img = Image.open(os.path.join(img_dir, img_name)).convert('RGB')
    img = img_prepr(img) # transform
    input_batch = img.unsqueeze(0) # create a mini-batch as expected by the model
    with torch.no_grad():
        output = model(input_batch)
        
    # Access the FC layer features
    fmaps = {}
    for i, (layer_name, feat_vals) in enumerate(all_feats.items()):
        if i == 0:
             fmaps[layer_name] = all_feats[layer_name]
        else:
             fmaps[layer_name] = torch.vstack((fmaps[layer_name], all_feats[layer_name])) # (num_img, num_feat 1000)

for layer in fmaps.keys():
    print(layer, fmaps[layer].shape)
print(len(fmaps.keys()))

# Save feature maps
# save_dir = f'dataset/sleemory_{args.dataset}/dnn_feature_maps'
# if os.path.isdir(save_dir) == False:
# 	os.makedirs(save_dir)
# scipy.io.savemat(f'{save_dir}/{DNNetworks}_fmaps.mat', {'fmaps': fmaps}) 