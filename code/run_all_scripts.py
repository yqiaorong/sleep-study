import os

# =============================================================================
# Preprocess all THINGS EEG2 raw data
# =============================================================================
for i in range(1,10):
    os.system(f'python code/THINGS_EEG2_preprocess.py --subj {i+1}')