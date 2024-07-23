# Sleep Study

## dataset

1. [THINGS EEG2 dataset](https://www.sciencedirect.com/science/article/pii/S1053811922008758?via%3Dihub)

    The download path:
   
* [Raw EEG data](https://osf.io/crxs4/): ../dataset/THINGS_EEG2/raw_data/

* [Image set](https://osf.io/y63gw/): ../dataset/THINGS_EEG2/image_set/

2. [THINGS EEG1 dataset](https://www.nature.com/articles/s41597-021-01102-7) 

    The download path:

* [THINGS EEG1](https://openneuro.org/datasets/ds003825/versions/1.2.0): ../dataset/THINGS_EEG1/

## code

The download path: ../code/

The order of running the scripts: 

### whiten_data/

* whiten_localiser.py --adapt_to [ /_sleemory] (This script whitens the sleemory localiser eeg and re-orders it according to the image order by stacking eegs with the same stimuli, whitening them and then taking the average as the final whitened eeg representing the corresponding image.)

* whiten_retrieval_re.py

### fmaps/

* Alexnet_full_fmaps.py --dataset [ localiser / retrieval ] --pretrained 
  
  Alexnet_best_fmaps.py --dataset [ localiser / retrieval ] --pretrained --num_feat

* CLIP_fmaps.py --dataset [ localiser / retrieval ]

* resnext_fmaps.py --dataset [ localiser / retrieval ]

* BLIP-2_capt.py --dataset [ localiser / retrieval ] (This script generates captions for images and extracts text features. )

### sleemory_localiser/

* build_enc_model.py --networks

* pred_eeg.py --networks

### sleemory_retrieval/



### THINGS_preprocess/

* preprocess_all.py --sfreq --adapt_to (This file preprocesses all THINGS EEG2 and EEG1 raw data. If it's adapted to sleemory, sfreq = 250, adapt_to = _sleemory )


### THINGS/

* 1_alexnet.py (This file extracts THINGS images features by Alexnet. )

* 2_img_feat_selection.py --num_feat --adapt_to [ /_sleemory] (This file selects the best N features of THINGS images features based on THINGS EEG2 training data.)

* 3_enc_model.py --num_feat --adapt_to [ /_sleemory] (This file trains the linear regression model on THINGS EEG2 training data. )

* 4_corr.py --test_dataset --num_feat (THINGS_EEG2/THINGS_EEG1) (This file tests the encoding model on either THINGS EEG2 test data or THINGS EEG1 data.)