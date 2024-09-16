import os

# subs = range(2, 27)
# for sub in subs:
#     if sub!=17:
#         os.system(f'python3 code/vox/gptneo_ridge_model_PCA2.py --sub {sub}')

subs = range(11, 27)
for sub in subs:
    if sub!=17:
        os.system(f'python3 code/vox/ResNet_ridge_model_PCA2.py --sub {sub} --layer_name fc')