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

DNNetworks = 'ResNet-fc'

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
    all_feats['fc'] = output
model.fc.register_forward_hook(hook_fn)

# Extract
img_dir = f'dataset/sleemory_{args.dataset}/image_set/'
fmaps = None
for img_name in tqdm(os.listdir(img_dir)):
    img = Image.open(os.path.join(img_dir, img_name)).convert('RGB')
    img = img_prepr(img) # transform
    input_batch = img.unsqueeze(0).to(device) # create a mini-batch as expected by the model
    with torch.no_grad():
        output = model(input_batch)
        
    # Access the FC layer features
    fc_output = all_feats['fc']   
    if fmaps is None:
        fmaps = fc_output
    else:
        fmaps = torch.vstack((fmaps, fc_output)) # (num_img, num_feat 1000)

# Save feature maps
save_dir = f'dataset/sleemory_{args.dataset}/dnn_feature_maps/full_feature_maps/{DNNetworks}/'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
scipy.io.savemat(f'{save_dir}/{DNNetworks}_fmaps.mat', {'fmaps': fmaps,
                                                        'imgs_all': os.listdir(img_dir)}) 