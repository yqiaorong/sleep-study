"""This script should be run after the full feature maps of GPTNEO and ResNet have been extracted. """
import os

# Specify the subjects 
subs = range(2, 26)

for sub in subs:

    # =============================================================================
    # GPTNEO
    # =============================================================================

    # Extract the best feature maps
    os.system(f'python3 code/fmaps_voxel/gptneo_best_fmaps_all_layers.py --sub {sub}')

    # Build the encoding model 
    os.system(f'python3 code/fmaps_voxel/build_enc_model.py --sub {sub} --networks GPTNEO --num_feat 1000')

    # Predict EEG
    os.system(f'python3 code/fmaps_voxel/pred_eeg.py --sub {sub} --networks GPTNEO --num_feat 1000')

    # =============================================================================
    # ResNet
    # =============================================================================

    # Extract the best feature maps per layers
    os.system(f'python3 code/fmaps_voxel/resnext_best_fmaps_chunkall.py --sub {sub}')

    # Extract the best feature maps
    os.system(f'python3 code/fmaps_voxel/resnext_best_fmaps_final.py --sub {sub}')

    # Build the encoding model 
    os.system(f'python3 code/fmaps_voxel/build_enc_model.py --sub {sub} --networks ResNet --num_feat 1000')

    # Predict EEG
    os.system(f'python3 code/fmaps_voxel/pred_eeg.py --sub {sub} --networks ResNet --num_feat 1000')