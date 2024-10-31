import pickle
import numpy as np
import mat73
import os
import scipy

def load_eeg_to_train_enc(sub, whiten = False):

	eeg_fname = f'sub-{sub:03d}_task-localiser_source_data'
	if whiten==False:
		eeg_dir = '/dataset/sleemory_localiser/preprocessed_data/'
		
		eeg_data = mat73.loadmat(os.path.join(eeg_dir, eeg_fname+'.mat'))
		eeg = eeg_data['sub_eeg_loc']['eeg'].astype(np.float32)
		print(f'Initial eeg shape {eeg.shape}')
		# eeg_vox = np.squeeze(eeg[:, vox_idx, :])
		# print(f'Initial eeg vox shape {eeg_vox.shape}')

		eeg_labels = eeg_data['sub_eeg_loc']['images']
		eeg_labels = [s[0] for s in eeg_labels]
		eeg_labels = np.asarray(eeg_labels)
		del eeg_data
	else:
		eeg_dir = 'output/sleemory_localiser_vox/whiten_eeg/'
		with open(os.path.join(eeg_dir, eeg_fname+'.pkl'), 'rb') as f:
			eeg_data = pickle.load(f)

			eeg = eeg_data['sub_eeg_loc']['eeg'].astype(np.float32)
			print(f'Initial eeg shape {eeg.shape}')
		
			# eeg = eeg_data['sub_eeg_loc/eeg']
			# eeg_labels = eeg_data['sub_eeg_loc/images']
			eeg_labels = eeg_data['sub_eeg_loc']['images']
			eeg_labels = [s[0] for s in eeg_labels]
			eeg_labels = np.asarray(eeg_labels)
			# print(eeg_labels)

	return eeg, eeg_labels

def customize_fmaps(eeg, eeg_labels, fmaps_all, flabels_all):

	# Check the order of two labels
	reorder_fmaps = np.empty((eeg.shape[0], fmaps_all.shape[1]))
	reorder_flabels = []

	for idx, eeg_label in enumerate(eeg_labels):
		# print(eeg_label)
		fmaps_idx = np.where(flabels_all == eeg_label)[0]
		# print(f'fmaps idx: {fmaps_idx}')
		# print(flabels_all[fmaps_idx])
		if eeg_label == flabels_all[fmaps_idx]:
			# print('fmaps idx correct')
			reorder_fmaps[idx] = fmaps_all[fmaps_idx]
			reorder_flabels.append(flabels_all[fmaps_idx])
		else:
			print('fmaps idx incorrect')
		# print('')
	# print(reorder_fmaps.shape, eeg.shape)
	# print(f'Is there nan in reordered fmaps? {np.isnan(reorder_fmaps).any()}')

	# for idx in range(eeg_labels.shape[0]):
	# 	print(eeg_labels[idx], reorder_flabels[idx])
	return reorder_fmaps, np.squeeze(reorder_flabels)

# Load the feature maps

def load_fmaps(dataset, network):
	fmaps_fname = f'{network}_fmaps.mat'
	print(fmaps_fname)
	fmaps_path = f'/dataset/sleemory_{dataset}/dnn_feature_maps/full_feature_maps/{network}/{fmaps_fname}'
	print(fmaps_path)
	fmaps_data = scipy.io.loadmat(fmaps_path)
	print('fmaps successfully loaded')

	# Load fmaps 
	fmaps = fmaps_data['fmaps'].astype(np.float32) # (img, 'num_token', num_feat)
	print(fmaps.shape)
	
	fmap_labels = fmaps_data['imgs_all'].squeeze()
	try:
		fmap_labels = np.char.rstrip(fmap_labels)
	except:
		pass
	print(fmap_labels.shape)
	return fmaps, fmap_labels

def load_GPTNeo_fmaps(dataset):
	fmaps_data = scipy.io.loadmat(f'dataset/sleemory_{dataset}/dnn_feature_maps/full_feature_maps/gptneo/gptneo_fmaps.mat')

	fmaps_labels = fmaps_data['imgs_all']
	fmaps_labels = [np.char.rstrip(s) for s in fmaps_labels] # remove extra spacing in strings
	fmaps_labels = np.array(fmaps_labels)

	# fmaps_capts  = fmaps_data['captions']
	for layer in fmaps_data.keys():
		# print(layer)
		if layer.startswith('layer'):
			# print(fmaps_data[layer].shape, 'take')
			if layer == 'layer_0_embeddings':
				fmaps = fmaps_data[layer]
			else:
				fmaps = np.concatenate((fmaps, fmaps_data[layer]), axis=1)
	print(f'fmaps all shape: {fmaps.shape}')
	del fmaps_data
	return fmaps, fmaps_labels

def load_AlexNet_fmaps(dataset, layer):
	fmaps_fname = f'AlexNet-{layer}_PCA_fmaps.mat'
	print(fmaps_fname)
	fmaps_path = f'dataset/sleemory_{dataset}/dnn_feature_maps/PCA_feature_maps/{fmaps_fname}'
	print(fmaps_path)
	fmaps_data = scipy.io.loadmat(fmaps_path)
	print('fmaps successfully loaded')

	# Load fmaps 
	fmaps = fmaps_data['fmaps'].astype(np.float32) # (img, 'num_token', num_feat)
	print(fmaps.shape)

	# load labels (contains .jpg)
	fmap_labels = np.char.rstrip(fmaps_data['imgs_all'])
	fmap_labels = np.array([item +'.jpg' for item in fmap_labels])
	print(fmap_labels.shape)

	return fmaps, fmap_labels

def load_and_match_eeg_and_fmaps(sub, args, fmaps, fmap_labels):

	# Load eeg
	print(f'sub {sub}')
	eeg, eeg_labels = load_eeg_to_train_enc(sub, whiten=args.whiten)

	# Reorder localiser fmaps
	reorder_fmaps, reorder_flabels = customize_fmaps(eeg, eeg_labels, fmaps, fmap_labels)

	print(eeg.shape, eeg_labels.shape)
	print(reorder_fmaps.shape, reorder_flabels.shape)

	return eeg, eeg_labels, reorder_fmaps, reorder_flabels
