import os

for i in range(50):
    os.system(f'python code/sleemory/4_corr_img.py --method pattern --img_cond_idx {i}')