"""This script should be run after the full feature maps of GPTNEO and ResNet have been extracted. """
import os

# Specify the subjects 
subs = range(2, 10)

# for sub in subs:
#     if sub != 17:
#         os.system(f'python3 code/whiten_data/whiten_localiser_voxel.py --sub {sub}')

#         # =============================================================================
#         # GPTNEO per sub per time
#         # =============================================================================

#         # Extract the best feature maps, build the encoding model and predict EEG
#         os.system(f'python3 code/vox/gptneo_enc_model_time.py --sub {sub}')

# =============================================================================
# ResNet fc and layer4.2.conv3 per sub per time
# =============================================================================

for sub in subs:
    if sub != 17:
        # Build the encoding model and predict EEG
        os.system(f'python3 code/vox/ResNet_layer_enc_model_time.py --sub {sub} --layer_name fc')
        os.system(f'python3 code/vox/ResNet_layer_enc_model_time.py --sub {sub} --layer_name layer4.2.conv3')

    # # =============================================================================
    # # ResNet
    # # =============================================================================

    # # Extract the best feature maps per layers
    # os.system(f'python3 code/fmaps_voxel/resnext_best_fmaps_chunkall.py --sub {sub}')

    # # Extract the best feature maps
    # os.system(f'python3 code/fmaps_voxel/resnext_best_fmaps_final.py --sub {sub}')

    # # Build the encoding model 
    # os.system(f'python3 code/fmaps_voxel/build_enc_model.py --sub {sub} --networks ResNet --num_feat 1000')

    # # Predict EEG
    # os.system(f'python3 code/fmaps_voxel/pred_eeg.py --sub {sub} --networks ResNet --num_feat 1000')