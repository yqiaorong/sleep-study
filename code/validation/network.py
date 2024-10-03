import os
networks = 'mpnet'
subs = range(2, 27)
for sub in subs:
    if sub != 17:
        os.system(f'python3 code/validation/{networks}_ridge_fracs_PCA.py --sub {sub}')

os.system(f'python3 code/validation/avg_corr_each_model.py --networks {networks}')