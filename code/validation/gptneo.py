import os

subs = range(2, 27)
for sub in subs:
    if sub != 17:
        os.system(f'python3 code/validation/gptneo_ridge_fracs_PCA.py --sub {sub}')