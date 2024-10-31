import os

os.system('python3 code/validation/permutation_test.py --networks mpnet')

os.system('python3 code/validation/permutation_test.py --networks ResNeXt --layer_name fc')
os.system('python3 code/validation/permutation_test.py --networks ResNeXt --layer_name maxpool')
os.system('python3 code/validation/permutation_test.py --networks ResNeXt --layer_name layer3')

os.system('python3 code/validation/permutation_test.py --networks CLIP')
os.system('python3 code/validation/permutation_test.py --networks BLIP-2')