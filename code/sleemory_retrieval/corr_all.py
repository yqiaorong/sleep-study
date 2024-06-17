import os

for sub in range(4, 26):
    if sub == 17:
        pass
    else:
        os.system(f'python code/sleemory_retrieval/corr_whitenFalse.py --sub {sub}')
        os.system(f'python code/sleemory_retrieval/corr_whitenTrue.py --sub {sub}')