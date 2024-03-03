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

* run_all_scripts.py (This file preprocesses all THINGS EEG2 and EEG1 raw data.)

* THINGS_alexnet.py (This file extracts THINGS images features by Alexnet. )

* image_feature_selection.py --num_feat (This file selects the best 300 features of THINGS images features based on THINGS EEG2 training data.)

* encoding_model.py --num_feat (This file trains the linear regression model on THINGS EEG2 training data. )

* correlation.py --test_dataset --num_feat  (THINGS_EEG2/THINGS_EEG1) (This file tests the encoding model on either THINGS EEG2 test data or THINGS EEG1 data.)