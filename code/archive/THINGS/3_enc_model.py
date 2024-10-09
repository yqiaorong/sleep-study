import os
import argparse
import pickle
from func import train_model_THINGS2

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--layer_name', default='conv5', type=str)
parser.add_argument('--num_feat', default=300, type=int)
parser.add_argument('--adapt_to', default='', type=str) # [/_sleemory]
args = parser.parse_args()

print('')
print(f'>>> Train the encoding model <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
 
# =============================================================================
# Train the encoding model 
# =============================================================================

reg = train_model_THINGS2(args) # depend on the num_feat

# Save the model
model_dir = os.path.join('dataset','THINGS_EEG2', 'model')
pickle.dump(reg, 
            open(os.path.join(model_dir, f'reg_model_{args.num_feat}{args.adapt_to}.pkl'), 
                'wb'))