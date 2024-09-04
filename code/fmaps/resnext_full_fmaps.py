import os
from tqdm import tqdm
import argparse

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None, type=str)
args = parser.parse_args()

 
img_dir = f'/home/simon/Documents/gitrepos/shannon_encodingmodelsEEG/dataset//sleemory_{args.dataset}/image_set/'
img_list = os.listdir(img_dir)
print(len(img_list))
for img_idx, img in enumerate(img_list):
    if not os.path.exists(f'dataset/sleemory_{args.dataset}/dnn_feature_maps/full_feature_maps/ResNet/{img}_feats.mat'):
        os.system(f'python code/fmaps/resnext_full_fmaps_single_img.py --dataset {args.dataset} --img_idx {img_idx}')