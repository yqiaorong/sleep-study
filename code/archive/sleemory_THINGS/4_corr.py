import os
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None, type=str)
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--layer_name', default='conv5', type=str)

parser.add_argument('--num_feat', default=1000, type=int)
parser.add_argument('--z_score', default=True, type=bool)

parser.add_argument('--method', default='img_cond', type=str) # [img_cond / pattern / pattern_all]
parser.add_argument('--img_cond_idx', default=-1, type=int) # for [pattern]
parser.add_argument('--pattern_all_range', default=[0, -1], nargs='+', type=int) # for [pattern_all]
args = parser.parse_args()

print('')
print(f'>>> Correlation of sleemory based on THINGS ({args.method}) <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
print('')

# =============================================================================
# Load data
# =============================================================================

load_dir = f'output/{args.dataset}_THINGS/test_eeg'

# test eeg
eeg = np.load(os.path.join(load_dir, 'whiten_test_eeg.npy'), allow_pickle=True).item()
test_eeg2 = eeg['test_eeg2']
del eeg
# pred eeg
pred_eeg = np.load(os.path.join(load_dir, f'z{args.z_score}_{args.num_feat}feat_pred_eeg.npy'), 
                   allow_pickle=True)

print('Final pred_eeg_data shape (img, ch, time)', pred_eeg.shape)
print('Final test_eeg_data shape (img, ch, time)', test_eeg2.shape)

# Set the params
t_THINGS = pred_eeg.shape[2]
t_sleemory = test_eeg2.shape[2]
num_ch = test_eeg2.shape[1]

# =============================================================================
# Save directory
# =============================================================================

save_dir = f'output/{args.dataset}_THINGS/enc_acc/enc acc ({args.method})/{args.num_feat}_feat'
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)

# The file name of saved enc acc figures
# plot_name1 = f'enc acc ({args.method}) M1 ({args.num_feat} feats) z scored {args.z_score}'
plot_name2 = f'enc acc ({args.method}) M2 ({args.num_feat} feats) z scored {args.z_score}'

# =============================================================================
# Calculate the encoding accuracy
# =============================================================================

if args.method == 'img_cond':
    
    # enc_acc = np.empty((num_ch, t_THINGS, t_sleemory))
    enc_acc2 = np.empty((num_ch, t_THINGS, t_sleemory))
                
    for ch in tqdm(range(num_ch), desc='Correlation (img patterns)'): # iterate over channels
        for t_TH in range(t_THINGS):
            TH_values = pd.Series(pred_eeg[:, ch, t_TH])
            for t_s in range(t_sleemory):
                # s_values1 = pd.Series(test_eeg[:, ch, t_s])
                s_values2 = pd.Series(test_eeg2[:, ch, t_s])
                # enc_acc[ch, t_TH, t_s] = TH_values.corr(s_values1)
                enc_acc2[ch, t_TH, t_s] = TH_values.corr(s_values2)
                
    # Average the encoding accuracy across channels
    # enc_acc = np.mean(enc_acc, 0)
    enc_acc2 = np.mean(enc_acc2, 0)
        
elif args.method == 'pattern':
    
    # enc_acc = np.empty((t_THINGS, t_sleemory))
    enc_acc2 = np.empty((t_THINGS, t_sleemory))
            
    for t_TH in tqdm(range(t_THINGS), desc='Correlation (EEG patterns)'):
        TH_values = pd.Series(pred_eeg[args.img_cond_idx, :, t_TH])
        for t_s in range(t_sleemory):
            # s_values1 = pd.Series(test_eeg[args.img_cond_idx, :, t_s])
            s_values2 = pd.Series(test_eeg2[args.img_cond_idx, :, t_s])
            # enc_acc[t_TH, t_s] = TH_values.corr(s_values1)
            enc_acc2[t_TH, t_s] = TH_values.corr(s_values2)
            
    # Change plot names
    # plot_name1 = f'{args.img_cond_idx:003d}' + plot_name1
    plot_name2 = f'{args.img_cond_idx:003d}' + plot_name2
    
elif args.method == 'pattern_all': # This one is so time consuming!
    
    num_img_cond = args.pattern_all_range[1] - args.pattern_all_range[0]
    
    # enc_acc = np.empty((num_img_cond, t_THINGS, t_sleemory))
    enc_acc2 = np.empty((num_img_cond, t_THINGS, t_sleemory))
    for idx, item in enumerate(tqdm(range(args.pattern_all_range[0], args.pattern_all_range[1]), 
                                    desc='Coorelation (EEG patterns)')):
        for t_TH in range(t_THINGS):
            TH_values = pd.Series(pred_eeg[item, :, t_TH])
            for t_s in range(t_sleemory):
                # s_values1 = pd.Series(test_eeg[item, :, t_s])
                s_values2 = pd.Series(test_eeg2[item, :, t_s])
                # enc_acc[idx, t_TH, t_s] = TH_values.corr(s_values1)
                enc_acc2[idx, t_TH, t_s] = TH_values.corr(s_values2)
               
    # Save the results
    enc_acc_result = {
                    # 'enc_acc': enc_acc, 
                    'enc_acc2': enc_acc2}
    with open(os.path.join(save_dir, 
                            f'enc_acc_{args.pattern_all_range[0]}-{args.pattern_all_range[1]}'
                            +f'_{args.num_feat}feats_z{args.z_score}'), 
                'wb') as f: 
        pickle.dump(enc_acc_result, f, protocol=4)
    
    # Average across img cond
    enc_acc2 = np.mean(enc_acc2, axis=0)

    # Change plot names
    # plot_name1 = f'{args.pattern_all_range[0]}_{args.pattern_all_range[1]}' + plot_name1
    plot_name2 = f'{args.pattern_all_range[0]}_{args.pattern_all_range[1]}' + plot_name2

print(f'The shape of encoding accuracy: {enc_acc2.shape}')

# =============================================================================
# Plot the correlation results
# =============================================================================
    
# # Plot all 2D results of method 1
# fig = plt.figure(figsize=(6, 5))
# im = plt.imshow(enc_acc, cmap='viridis',
# 				extent=[-0.2, 0.8, -0.25, 1], 
#                 origin='lower', aspect='auto')
# cbar = plt.colorbar(im)
# cbar.set_label('Values')
# # Plot borders
# plt.plot([-0.2, 0.8], [0,0], 'k--', lw=0.4)
# plt.plot([0,0], [-0.25, 1], 'k--', lw=0.4)
# plt.xlim([-0.2, 0.8])
# plt.ylim([-0.25, 1])
# plt.xlabel('THINGS time / s')
# plt.ylabel('Sleemory time / s')
# plt.title(f'Encoding accuracy ({args.method}) with {args.num_feat} features')
# fig.tight_layout()
# plt.savefig(os.path.join(save_dir, plot_name1))

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