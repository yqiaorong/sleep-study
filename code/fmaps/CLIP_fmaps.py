import torch
import clip
from PIL import Image
import os
import scipy
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
from torchvision import transforms as trn

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None, type=str)
args = parser.parse_args()

DNNetworks = 'CLIP'

print('')
print(f'>>> Extract sleemory images feature maps ({DNNetworks}) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
 


# Define the transform for images (Same as Alexnet)
size = 224
img_prepr = trn.Compose([
	trn.Resize((size,size)),
	trn.ToTensor(),
	trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



# =============================================================================
# Create dataset
# =============================================================================

# Customise dataset
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_labels = os.listdir(img_dir)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# Load images and convert to dataset
img_dir = f'dataset/sleemory_{args.dataset}/image_set/'
img_dataset = CustomImageDataset(img_dir=img_dir, transform=img_prepr)



# =============================================================================
# Feature maps
# =============================================================================

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load('ViT-B/32', device)

# Extract feature maps
def get_features(dataset, size):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=size)):
            features = model.encode_image(images.to(device))
            all_features.append(features)
            all_labels.append(labels)
    
    return torch.cat(all_features).cpu().numpy()

all_fmaps = get_features(img_dataset, size)

# Save feature maps
save_dir = f'dataset/sleemory_{args.dataset}/dnn_feature_maps'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
scipy.io.savemat(f'{save_dir}/{DNNetworks}_fmaps.mat', {'fmaps': all_fmaps}) 