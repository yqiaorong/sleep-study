import torch
from PIL import Image
from torchvision import transforms as trn
import os
import argparse
import scipy

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',    default=None,     type=str)
parser.add_argument('--layer_name', default='layer3', type=str) # layer3 / fc
args = parser.parse_args()

DNNetworks = f'ResNeXt-{args.layer_name}'

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
    all_feats[args.layer_name] = output
if args.layer_name == 'fc':
    model.fc.register_forward_hook(hook_fn)
elif args.layer_name == 'maxpool':
    model.maxpool.register_forward_hook(hook_fn)
elif args.layer_name == 'layer3':
    model.layer3.register_forward_hook(hook_fn)
     
# Extract
fmaps = None
flabels = []

img_dir = f'/dataset/sleemory_{args.dataset}/image_set/'

# Iterate over all images
for img_name in os.listdir(img_dir):
    if img_name.endswith(".jpg") or img_name.endswith(".JPEG"):
        print(img_name)
        flabels.append(img_name)

        img = Image.open(os.path.join(img_dir, img_name)).convert('RGB')
        img = img_prepr(img) # transform
        input_batch = img.unsqueeze(0).to(device) # create a mini-batch as expected by the model
        with torch.no_grad():
            output = model(input_batch)
            
        # Access the FC layer features
        layer_output = all_feats[args.layer_name]   
        if fmaps is None:
            fmaps = layer_output
        else:
            fmaps = torch.vstack((fmaps, layer_output)) # (num_img, num_feat 1000)
fmaps = fmaps.cpu()

fmaps = fmaps.reshape(fmaps.shape[0], -1)
print(fmaps.shape)

# Save feature maps
save_dir = f'dataset/sleemory_{args.dataset}/dnn_feature_maps/full_feature_maps/{DNNetworks}/'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
scipy.io.savemat(f'{save_dir}/{DNNetworks}_fmaps.mat', {'fmaps': fmaps,
                                                        'imgs_all': flabels}) 