import os

# =============================================================================
# Preprocess all THINGS EEG2 raw data
# =============================================================================
for i in range(10):
    os.system(f'python code/THINGS_EEG2_preprocess.py --subj {i+1}')
    
# =============================================================================
# Preprocess all THINGS EEG1 raw data
# =============================================================================
for i in range(1):
    os.system(f'python code/THINGS_EEG1_preprocess.py --subj {i+1}')