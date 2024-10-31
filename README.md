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

### fmaps/

Generate image captions using [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b):

* BLIP-2_capt.py --dataset [ localiser / retrieval ] (This script generates captions for images.)

Then we manually adjusted the automatically generated captions. 

Extracts the caption text features from [BLIP-2](https://huggingface.co/Salesforce/blip2-opt-2.7b) / [CLIP](https://huggingface.co/openai/clip-vit-base-patch32):

* LLM_text_feats.py --dataset [ localiser / retrieval ] --networks [BLIP-2 / CLIP] 

Extract the caption text features from [GPT-Neo](https://huggingface.co/EleutherAI/gpt-neo-1.3B):

* gptneo_full_fmaps.py --dataset [ localiser / retrieval ]

Extract the caption text features from MPNet:

* mpnet_full_fmaps.py --dataset [ localiser / retrieval ]

Extracts the visual image features from ResNeXt [resnext101_32x8d_wsl](https://pytorch.org/hub/facebookresearch_WSL-Images_resnext/):

* resnext_layer_fmaps.py --dataset [ localiser / retrieval ] --layer_name [ maxpool / layer3 / fc] 

Extracts the visual image features from AlexNet and use PCA to select the best features:

* Alexnet_full_fmaps.py --pretrained --dataset [ localiser / retrieval ]

* Alexnet_best_fmaps_PCA.py --pretrained --layer [ conv'x' | x = 1, ...5 / fc'y' | y = 6, 7, 8 / all]

### validation/

Validation test of features from BLIP-2 / CLIP / AlexNet / ResNeXt:  

* valid_all.py 

  * model_ridge_fracs_PCA.py

  * avg_corr_each_model.py

Permutation test of features from BLIP-2 / CLIP / ResNeXt:  

* permu_all.py 

  * permutation_test.py

Predict EEG from neural networks features using encoding models:

* pred_all.py
  
  * pred_retrieval_eeg.py

## Data structure

Neural Networks features

dataset
   |
   └───sleemory_localiser───dnn_feature_maps───full_feature_maps───mpnet

   └───sleemory_retrieval───dnn_feature_maps───full_feature_maps
