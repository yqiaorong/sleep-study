import os
import scipy
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from func import mvnn_mean, mvnn
from matplotlib import pyplot as plt
from scipy.stats import pearsonr as corr

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--layer_name', default='conv5', type=str)

parser.add_argument('--num_feat', default=1000, type=int)
parser.add_argument('--z_score', default=True, type=bool)

parser.add_argument('--method', default='img_cond', type=str) # [img_cond / pattern / pattern_all]
parser.add_argument('--img_cond_idx', default=-1, type=int) # for [pattern]
parser.add_argument('--pattern_all_range', default=[0, -1], nargs='+', type=int) # for [pattern_all]
args = parser.parse_args()

print('')
print(f'>>> Correlation of sleemory ({args.method}) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')

# =============================================================================
# Load data
# =============================================================================

load_path = f'output/sleemory/test_eeg/z{args.z_score}_{args.num_feat}feat.npy'

load_eeg = np.load(load_path, allow_pickle=True).item()
test_eeg = load_eeg['test_eeg']
test_eeg2 = load_eeg['test_eeg2']
pred_eeg = load_eeg['pred_eeg']

print('Final pred_eeg_data shape (img, ch, time)', pred_eeg.shape)
print('Final test_eeg_data shape (img, ch, time)', test_eeg.shape, test_eeg2.shape)

# Set the params
t_THINGS = pred_eeg.shape[2]
t_sleemory = test_eeg.shape[2]
num_ch = test_eeg.shape[1]

# =============================================================================
# Save directory
# =============================================================================

save_dir = f'output/sleemory/enc_acc/enc acc ({args.method})/{args.num_feat}_feat'
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)

# The file name of saved enc acc figures
plot_name1 = f'enc acc ({args.method}) M1 ({args.num_feat} feats) z scored {args.z_score}'
plot_name2 = f'enc acc ({args.method}) M2 ({args.num_feat} feats) z scored {args.z_score}'

# =============================================================================
# Calculate the encoding accuracy
# =============================================================================

if args.method == 'img_cond':
    
    enc_acc = np.empty((num_ch, t_THINGS, t_sleemory))
    enc_acc2 = np.empty((num_ch, t_THINGS, t_sleemory))
    for ch in tqdm(range(num_ch), desc='Correlation (img patterns)'): # iterate over channels
        for t_s in range(t_sleemory):
            for t_TH in range(t_THINGS):
                enc_acc[ch, t_TH, t_s] = corr(pred_eeg[:, ch, t_TH], test_eeg[:, ch, t_s])[0]
                enc_acc2[ch, t_TH, t_s] = corr(pred_eeg[:, ch, t_TH], test_eeg2[:, ch, t_s])[0]
    # Average the encoding accuracy across channels
    enc_acc = np.mean(enc_acc, 0)
    enc_acc2 = np.mean(enc_acc2, 0)
        
elif args.method == 'pattern':
    
    enc_acc = np.empty((t_THINGS, t_sleemory))
    enc_acc2 = np.empty((t_THINGS, t_sleemory))
    for t_s in tqdm(range(t_sleemory), desc='Correlation (EEG patterns)'):
        for t_TH in range(t_THINGS):
            enc_acc[t_TH, t_s] = corr(pred_eeg[args.img_cond_idx, :, t_TH],
                                        test_eeg[args.img_cond_idx, :, t_s])[0]
            enc_acc2[t_TH, t_s] = corr(pred_eeg[args.img_cond_idx, :, t_TH],
                                        test_eeg2[args.img_cond_idx, :, t_s])[0]
            
    # Change plot names
    plot_name1 = f'{args.img_cond_idx:003d}' + plot_name1
    plot_name2 = f'{args.img_cond_idx:003d}' + plot_name2
    
elif args.method == 'pattern_all': # This one is so time consuming!
    
    num_img_cond = args.pattern_all_range[1] - args.pattern_all_range[0]
    
    enc_acc = np.empty((num_img_cond, t_THINGS, t_sleemory))
    enc_acc2 = np.empty((num_img_cond, t_THINGS, t_sleemory))
    for idx, item in enumerate(tqdm(range(args.pattern_all_range[0], args.pattern_all_range[1]), 
                                    desc='Coorelation (EEG patterns)')):
        for t_s in range(t_sleemory):
            for t_TH in range(t_THINGS):
                enc_acc[idx, t_TH, t_s] = corr(pred_eeg[idx, :, t_TH], test_eeg[idx, :, t_s])[0]
                enc_acc2[idx, t_TH, t_s] = corr(pred_eeg[idx, :, t_TH], test_eeg2[idx, :, t_s])[0]
               
    # Save the results
    enc_acc_result = {'enc_acc': enc_acc, 'enc_acc2': enc_acc2}
    with open(os.path.join(save_dir, 
                           f'enc_acc_{args.pattern_all_range[0]}_{args.pattern_all_range[1]}'
                           +f'{args.num_feat}feats_z{args.z_score}'), 
              'wb') as f: 
        pickle.dump(enc_acc_result, f, protocol=4)
        
    # Average across img cond
    enc_acc = np.mean(enc_acc, axis=0)
    enc_acc2 = np.mean(enc_acc2, axis=0)
    
    # Change plot names
    plot_name1 = f'{args.pattern_all_range[0]}_{args.pattern_all_range[1]}' + plot_name1
    plot_name2 = f'{args.pattern_all_range[0]}_{args.pattern_all_range[1]}' + plot_name2
    
print(f'The shape of encoding accuracy: {enc_acc.shape}, {enc_acc2.shape}')

# =============================================================================
# Plot the correlation results
# =============================================================================
    
# Plot all 2D results of method 1
fig = plt.figure(figsize=(6, 5))
im = plt.imshow(enc_acc, cmap='viridis',
				extent=[-0.2, 0.8, -0.25, 1], 
                origin='lower', aspect='auto')
cbar = plt.colorbar(im)
cbar.set_label('Values')
# Plot borders
plt.plot([-0.2, 0.8], [0,0], 'k--', lw=0.4)
plt.plot([0,0], [-0.25, 1], 'k--', lw=0.4)
plt.xlim([-0.2, 0.8])
plt.ylim([-0.25, 1])
plt.xlabel('THINGS time / s')
plt.ylabel('Sleemory time / s')
plt.title(f'Encoding accuracy ({args.method}) with {args.num_feat} features')
fig.tight_layout()
plt.savefig(os.path.join(save_dir, plot_name1))

# Plot all 2D results of method 2
fig = plt.figure(figsize=(6, 5))
im = plt.imshow(enc_acc2, cmap='viridis',
				extent=[-0.2, 0.8, -0.25, 1], 
                origin='lower', aspect='auto')
cbar = plt.colorbar(im)
cbar.set_label('Values')
# Plot borders
plt.plot([-0.2, 0.8], [0,0], 'k--', lw=0.4)
plt.plot([0,0], [-0.25, 1], 'k--', lw=0.4)
plt.xlim([-0.2, 0.8])
plt.ylim([-0.25, 1])
plt.xlabel('THINGS time / s')
plt.ylabel('Sleemory time / s')
plt.title(f'Encoding accuracy ({args.method}) with {args.num_feat} features')
fig.tight_layout()
plt.savefig(os.path.join(save_dir, plot_name2))