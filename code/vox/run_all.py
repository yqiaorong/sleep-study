import os

subs = range(3, 27)
# for sub in subs:
#     if sub!=17:
#         os.system(f'python3 code/vox/gptneo_ridge_model_PCA2.py --sub {sub}')

# for sub in subs:
#     if sub!=17:
#         os.system(f'python3 code/vox/ResNet_ridge_model_PCA2.py --sub {sub} --layer_name fc')

for sub in subs:
    if sub!=17:
        os.system(f'python3 code/vox/mpnet_ridge_fracs_PCA.py --sub {sub}')