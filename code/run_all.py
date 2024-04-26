import os
img_range = [100, 200, 300, 400, 500, 600, 700, 800, 864]
for i in range(8):
    os.system(f'python code/sleemory/4_corr.py --num_feat -1 --method pattern_all --pattern_all_range {img_range[i]} {img_range[i+1]}')