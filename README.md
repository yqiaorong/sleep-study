# Sleep Study

## dataset

1. [THINGS EEG2 dataset](https://www.sciencedirect.com/science/article/pii/S1053811922008758?via%3Dihub)

The download path:
   
* [Raw EEG data](https://osf.io/crxs4/): ../dataset/THINGS_EEG2/raw_data/

* [Image set](https://osf.io/y63gw/): ../dataset/THINGS_EEG2/image_set/


## code

The download path: ../code/

The order of running the scripts: 

* run_all_scripts.py (This file preprocesses all THINGS EEG2 raw data.)

* THINGS_alexnet.py (This file extracts THINGS images features by Alexnet. )

* image_feature_selection.py (This file selects the best 300 features of THINGS images features.)

* encoding_model.py (This file trains the linear regression model on THINGS EEG2. )

* correlation.py (This file tests the encoding model.)