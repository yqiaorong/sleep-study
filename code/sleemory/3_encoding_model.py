import os
import pickle
import argparse
from func import train_model_THINGS2

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--layer_name', default='conv5', type=str)
parser.add_argument('--num_feat', default=300, type=int)
args = parser.parse_args()

print('')
print(f'>>> Train the encoding model on THINGS EEG2 adapted to sleemory <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
 
# =============================================================================
# Train the encoding model 
# =============================================================================

reg = train_model_THINGS2(args)

# Save the model
save_dir = os.path.join('dataset','temp_sleemory')
pickle.dump(reg, open(os.path.join(save_dir, f'reg_model_{args.num_feat}.pkl'), 'wb'))