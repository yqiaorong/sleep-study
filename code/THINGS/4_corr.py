import os
import pickle
import argparse
import numpy as np
import pingouin as pg
from tqdm import tqdm
from matplotlib import pyplot as plt
from func import load_full_fmaps, corr

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--test_dataset',default='THINGS_EEG2',type=str)
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--layer_name', default='conv5', type=str)
parser.add_argument('--num_feat', default=300, type=int)
args = parser.parse_args()

print('')
print(f'>>> Test the encoding model on {args.test_dataset} <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
 
# =============================================================================
# Load the full test feature maps
# =============================================================================

fmaps_test = load_full_fmaps(args, 'test')

# =============================================================================
# Load the feature selection model and apply feature selection to test fmaps
# =============================================================================

model_path = 'dataset/THINGS_EEG2'
feat = pickle.load(open(os.path.join(model_path, 
                                     f'feat_model_{args.num_feat}.pkl'), 'rb'))
best_feat_test = feat.transform(fmaps_test[args.layer_name])
print(f'The new test fmaps shape {best_feat_test.shape}')

# =============================================================================
# Load the encoding model
# =============================================================================

reg = pickle.load(open(os.path.join(model_path, 
                                    f'reg_model_{args.num_feat}.pkl'), 'rb'))

# =============================================================================
# Load the test EEG data and predict EEG from fmaps
# =============================================================================

# Load the test EEG data directory
eeg_train_dir = os.path.join('dataset', args.test_dataset, 'preprocessed_data')
# Iterate over subjects
eeg_data_test = []

if args.test_dataset == 'THINGS_EEG1':
    subj_range = [1,3]
elif args.test_dataset == 'THINGS_EEG2':
    subj_range = [1,11]
num_subj = subj_range[1]-subj_range[0]

for test_subj in tqdm(range(subj_range[0], subj_range[1]), 
                            desc=f'load {args.test_dataset} test data'):
    # Load the THINGS2 training EEG data
    data = np.load(os.path.join(eeg_train_dir,'sub-'+format(test_subj,'02'),
                  'preprocessed_eeg_test.npy'), allow_pickle=True).item()
    # Average the training EEG data across repetitions: (200,64,100)
    data = np.mean(data['preprocessed_eeg_data'], 1)
    # Drop the sim channel of THINGS EEG2: (200,63,100)
    if args.test_dataset == 'THINGS_EEG2':
        data = np.delete(data, -1, axis=1)
    # Average the training EEG data over time: (200,63)
    data = np.mean(data, -1)
    # Average the training EEG data across electrodes: (200,)
    data = np.mean(data, -1)
    # Append individual data
    eeg_data_test.append(data)
    del data
# Average the training EEG data across subjects: (200,)
eeg_data_train = np.mean(eeg_data_test, 0)

# Predict the test EEG data 
pred_eeg = reg.predict(best_feat_test)
print('pred_eeg_data_test shape', pred_eeg.shape)

# =============================================================================
# Get the encoding accuracy
# =============================================================================

# Get the encoding accuracy for each subject
tot_accuracy = np.empty((num_subj,100))
for i, test_subj in enumerate(tqdm(range(subj_range[0], subj_range[1]), 
                            desc=f'{args.test_dataset} correlation')):
    accuracy, times = corr(args, pred_eeg, test_subj)
    tot_accuracy[i] = accuracy

# =============================================================================
# Plot the correlation results
# =============================================================================
    
# Create the saving directory
save_dir = os.path.join('output', f'{args.test_dataset}')
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)

### All subjects ###
plt.figure(1)
plt.plot([-.2, .8], [0, 0], 'k--', [0, 0], [-1, 1], 'k--')
# Set the plot colour spectum
cmap = "cividis"
colours = plt.colormaps[cmap](np.linspace(0,1,num_subj))
# Plot
for i in range(num_subj):
    plt.plot(times, tot_accuracy[i], color = colours[i], alpha=0.2)
plt.plot(times, np.mean(tot_accuracy,0), color='k', label='Correlation scores')
plt.xlabel('Time (s)')
plt.xlim(left=-.2, right=.8)
plt.ylabel('Pearson\'s $r$')
plt.ylim(bottom=-.1, top=.3)
plt.title(f'Encoding accuracy on {args.test_dataset} (Alexnet): num_feat {args.num_feat}')
plt.legend(loc='best')
plt.savefig(os.path.join(save_dir, 
            f'Encoding accuracy on {args.test_dataset} (Alexnet) num_feat {args.num_feat}.jpg'))

### Average with confidence interval ###
# Set random seed for reproducible results
seed = 20200220
# Set the confidence interval
ci = np.empty((2,len(times)))
# Plot
plt.figure(2)
plt.plot([-.2, .8], [0, 0], 'k--', [0, 0], [-1, 1], 'k--')
# Calculate the confidence interval
for i in range(len(times)):
    ci[:,i] = pg.compute_bootci(tot_accuracy[:,i], func='mean', seed=seed)
# Plot the results with confidence interval
plt.plot(times, np.mean(tot_accuracy,0), color='salmon', 
         label='correlation mean scores with 95 \% confidence interval')
plt.fill_between(times, np.mean(tot_accuracy,0), ci[0], color='salmon', alpha=0.2)
plt.fill_between(times, np.mean(tot_accuracy,0), ci[1], color='salmon', alpha=0.2)
plt.xlabel('Time (s)')
plt.xlim(left=-.2, right=.8)
plt.ylabel('Pearson\'s $r$')
plt.ylim(bottom=-.05, top=.3)
plt.title(f'Averaged encoding accuracy on {args.test_dataset} (Alexnet): num_feat {args.num_feat}')
plt.legend(loc='best')
plt.savefig(os.path.join(save_dir, 
            f'Averaged encoding accuracy on {args.test_dataset} (Alexnet) num_feat {args.num_feat}.jpg'))