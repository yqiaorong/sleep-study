import os

os.system('python code/THINGS/3_enc_model.py --num_feat 1000 --adapt_to _sleemory')

os.system('python code/THINGS/2_img_feat_selection.py --num_feat -1 --adapt_to _sleemory')
os.system('python code/THINGS/3_enc_model.py --num_feat -1 --adapt_to _sleemory')

os.system('python code/sleemory/2_img_feat_selection.py --num_feat -1')
os.system('python code/sleemory/2_img_feat_selection.py --num_feat 1000')

os.system('python code/sleemory/4_corr.py --num_feat -1')
os.system('python code/sleemory/4_corr.py --num_feat 1000')