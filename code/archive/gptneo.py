import os

sub = 2 

# os.system(f'python3 code/vox/gptneo_enc_model_PCA.py --sub {sub} --whiten True')
# os.system(f'python3 code/vox/gptneo_enc_model_PCA.py --sub {sub}')

os.system(f'python3 code/vox/gptneo_enc_model_time.py --sub {sub} --whiten True')
os.system(f'python3 code/vox/gptneo_enc_model_time.py --sub {sub}')

os.system(f'python3 code/vox/gptneo_enc_model.py --whiten True')
os.system(f'python3 code/vox/gptneo_enc_model.py')