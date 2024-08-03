import os
from tqdm import tqdm
import argparse

# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None, type=str)
args = parser.parse_args()

 
img_dir = f'dataset/sleemory_{args.dataset}/image_set/'
img_list = os.listdir(img_dir)
for img_idx in tqdm(range(len(img_list))):
    os.system(f'python code/fmaps/resnext_full_fmaps_single_img.py --dataset {args.dataset} --img_idx {img_idx}')