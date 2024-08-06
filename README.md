# Sleep Study

## dataset

1. [THINGS EEG2 dataset](https://www.sciencedirect.com/science/article/pii/S1053811922008758?via%3Dihub)

    The download path:
   
* [Raw EEG data](https://osf.io/crxs4/): ../dataset/THINGS_EEG2/raw_data/

* [Image set](https://osf.io/y63gw/): ../dataset/THINGS_EEG2/image_set/

2. [THINGS EEG1 dataset](https://www.nature.com/articles/s41597-021-01102-7) 

    The download path:

* [THINGS EEG1](https://openneuro.org/datasets/ds003825/versions/1.2.0): ../dataset/THINGS_EEG1/

3. sleemory_localiser: 
  
  image_set: ../dataset/sleemory_localiser/image_set

  preprocessed_data: ../dataset/sleemory_localiser/preprocessed_data

4. sleemory_retrieval: 
  
  image_set: ../dataset/sleemory_localiser/image_set

  preprocessed_data: ../dataset/sleemory_localiser/preprocessed_data

## code

The download path: ../code/

The order of running the scripts: 

### whiten_data/

* whiten_localiser.py --adapt_to [ /_sleemory] --whiten (This script whitens the sleemory localiser eeg and re-orders it according to the image order by stacking eegs with the same stimuli, whitening them (optional) and then taking the average as the final whitened eeg representing the corresponding image.)

* whiten_retrieval_re.py

### fmaps/

Alexnet

* Alexnet_full_fmaps.py --dataset [ sleemory_localiser / sleemory_retrieval ] --pretrained 
  
* Alexnet_best_fmaps_each_layer.py --dataset [ sleemory_localiser / sleemory_retrieval ] --pretrained --num_feat --whiten

* Alexnet_best_fmaps_all_layer.py --dataset [ sleemory_localiser / sleemory_retrieval ] --pretrained --num_feat --whiten

CLIP

* CLIP_fmaps.py --dataset [ localiser / retrieval ]

ResNet

* resnext_fc_layer_fmaps.py --dataset [ localiser / retrieval ]

* resnext_full_fmaps.py --dataset [ localiser / retrieval ]
   
  resnext_full_fmaps_single_img.py --dataset [ localiser / retrieval ] --img_idx

* resnext_best_fmaps_chunkall.py --num_feat --whiten
  
  resnext_best_fmaps_chunk.py --layer_start_idx --num_feat --whiten

* resnext_best_fmaps_final.py --old_num_feat --new_num_feat --whiten

BLIP-2

* BLIP-2_capt.py --dataset [ localiser / retrieval ] (This script generates captions for images and extracts text features from automatically generated texts.)

* BLIP-2_text_feats.py --dataset [ localiser / retrieval ] --text_type (This script extracts the text features from either image names or the filtered generated texts.)

GPT-Neo

* gptneo_last_layer_fmaps.py --dataset [ localiser / retrieval ] 

* gptneo_full_fmaps.py --dataset [ localiser / retrieval ] 

* gptneo_best_fmaps_all_layer.py --dataset [ localiser / retrieval ] --num_feat --whiten

### sleemory_localiser/

* build_enc_model.py --networks --num_feat (countain additional prompt of 'whiten')

* pred_eeg.py --networks --num_feat --whiten

### sleemory_retrieval/



### THINGS_preprocess/

* preprocess_all.py --sfreq --adapt_to (This file preprocesses all THINGS EEG2 and EEG1 raw data. If it's adapted to sleemory, sfreq = 250, adapt_to = _sleemory )

### THINGS/

* 1_alexnet.py (This file extracts THINGS images features by Alexnet. )

* 2_img_feat_selection.py --num_feat --adapt_to [ /_sleemory] (This file selects the best N features of THINGS images features based on THINGS EEG2 training data.)

* 3_enc_model.py --num_feat --adapt_to [ /_sleemory] (This file trains the linear regression model on THINGS EEG2 training data. )

* 4_corr.py --test_dataset --num_feat (THINGS_EEG2/THINGS_EEG1) (This file tests the encoding model on either THINGS EEG2 test data or THINGS EEG1 data.)