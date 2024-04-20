import os
import pickle
import argparse
import numpy as np
from matplotlib import pyplot as plt

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--num_feat', default=1000, type=int)
parser.add_argument('--z_score', default=True, type=bool)
args = parser.parse_args()

load_dir = 'output/sleemory/enc_acc/enc acc (pattern_all)'
load_list = os.listdir(load_dir)

# Get the list of data file paths
data_fpaths = []
for fname in load_list:
    if not fname.endswith('png') and f'{args.num_feat}feats_z{args.z_score}' in fname:
        data_fpaths.append(os.path.join(load_dir, fname))

# Load all the data
for idx, fpath in enumerate(data_fpaths):
    data = np.load(fpath, allow_pickle=True)
        
    enc_acc = data['enc_acc']
    enc_acc2 = data['enc_acc2']
    
    if idx == 0:
        tot_enc_acc = enc_acc
        tot_enc_acc2 = enc_acc2
    else:
        tot_enc_acc = np.concatenate([tot_enc_acc, enc_acc], axis=0)
        tot_enc_acc2 = np.concatenate([tot_enc_acc2, enc_acc2], axis=0)

# Average all the data
tot_enc_acc = np.mean(tot_enc_acc, axis=0)
tot_enc_acc2 = np.mean(tot_enc_acc2, axis=0)

# =============================================================================
# Plot the overall correlation results
# =============================================================================

# Save dir
save_dir = 'output/sleemory/enc_acc'
plot_name1 = f'enc acc (pattern_all) M1 ({args.num_feat} feats) z scored True'
plot_name2 = f'enc acc (pattern_all) M2 ({args.num_feat} feats) z scored True'

# Plot all 2D results of method 1
fig = plt.figure(figsize=(6, 5))
im = plt.imshow(tot_enc_acc, cmap='viridis',
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
plt.title(f'Encoding accuracy (pattern_all) with {args.num_feat} features')
fig.tight_layout()
plt.savefig(os.path.join(save_dir, plot_name1))

# Plot all 2D results of method 2
fig = plt.figure(figsize=(6, 5))
im = plt.imshow(tot_enc_acc2, cmap='viridis',
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
plt.title(f'Encoding accuracy (pattern_all) with {args.num_feat} features')
fig.tight_layout()
plt.savefig(os.path.join(save_dir, plot_name2))

# Save data
saved_data = {'tot_enc_acc': tot_enc_acc, 'tot_enc_acc2': tot_enc_acc2}
with open(os.path.join(save_dir, 
                        f'enc_acc_pattern_all_'
                        +f'{args.num_feat}feats_z{args.z_score}'), 
          'wb') as f: 
    pickle.dump(saved_data, f, protocol=4)