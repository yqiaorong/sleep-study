"""Extract and save the AlexNet feature maps of sleemory images.

Parameters
----------
pretrained : bool
	If True use a pretrained network, if False a randomly initialized one.
layer_name : str
"""

import argparse
from torchvision import models
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable as V
from torchvision import transforms as trn
import os
from tqdm import tqdm
from PIL import Image

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--layer_name', default='conv5', type=str)
args = parser.parse_args()

print('')
print('Extract sleemory images feature maps AlexNet <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)

# =============================================================================
# Select the layers of interest and import the model
# =============================================================================
# Lists of AlexNet convolutional and fully connected layers
conv_layers = ['conv1', 'ReLU1', 'maxpool1', 'conv2', 'ReLU2', 'maxpool2',
	'conv3', 'ReLU3', 'conv4', 'ReLU4', 'conv5', 'ReLU5', 'maxpool5']

class AlexNet(nn.Module):
    def __init__(self, weights='IMAGENET1K_V1'):
        """Select the desired layers and create the model."""
        super(AlexNet, self).__init__()
        self.feat_list = [args.layer_name]
        self.alex_feats = models.alexnet(weights=weights).features

    def forward(self, x):
        """Extract the feature maps."""
        features = []
        for name, layer in self.alex_feats._modules.items():
            x = layer(x)
            if conv_layers[int(name)] in self.feat_list:
                features.append(x)
        return features
    
model = AlexNet()
if torch.cuda.is_available():
	model.cuda()
model.eval()

# =============================================================================
# Define the image preprocessing
# =============================================================================
centre_crop = trn.Compose([
	trn.Resize((224,224)),
	trn.ToTensor(),
	trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =============================================================================
# Load the sleemory images and extract the corresponding feature maps
# =============================================================================

# The main image directory
img_set_dir = os.path.join('dataset','temp_sleemory','image_set')
image_list = []
for root, _, files in os.walk(img_set_dir):
	for file in files:
		if file.endswith(".jpg") or file.endswith(".JPEG"):
			image_list.append(os.path.join(root,file))
image_list.sort()

# Create the saving directory if not existing
save_dir = os.path.join('dataset','temp_sleemory','dnn_feature_maps',
						'full_feature_maps','alexnet',
						'pretrained-'+str(args.pretrained))
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

# Extract and save the feature maps
for i, image in enumerate(tqdm(image_list, desc='extract fmaps')):
	img = Image.open(image).convert('RGB')
	input_img = V(centre_crop(img).unsqueeze(0))
	if torch.cuda.is_available():
		input_img=input_img.cuda()
	x = model.forward(input_img)
	feats = {}
	for f, feat in enumerate(x):
		feats[model.feat_list[f]] = feat.data.cpu().numpy()
	file_name = 'images_' + format(i+1, '04')
	np.save(os.path.join(save_dir, file_name), feats) 