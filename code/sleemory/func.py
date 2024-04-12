def load_full_fmaps(args, p='training'):
	import os
	import numpy as np
	from tqdm import tqdm
 
	feats = []
	final_fmaps = {}
	# The directory of the dnn training feature maps
	fmaps_dir = os.path.join('dataset','THINGS_EEG2','dnn_feature_maps',
							'full_feature_maps', 'Alexnet', 
							'pretrained-'+str(args.pretrained),
							f'{p}_images')
	fmaps_list = os.listdir(fmaps_dir)
	fmaps_list.sort()
	for f, fmaps in enumerate(tqdm(fmaps_list, desc='load THINGS training images')):
		fmaps_data = np.load(os.path.join(fmaps_dir, fmaps),
								allow_pickle=True).item()
		all_layers = fmaps_data.keys()
		for l, dnn_layer in enumerate(all_layers):
			if f == 0:
				feats.append([[np.reshape(fmaps_data[dnn_layer], -1)]])
			else:
				feats[l].append([np.reshape(fmaps_data[dnn_layer], -1)])
		
	final_fmaps[args.layer_name] = np.squeeze(np.asarray(feats[l]))
	print(f'The original {p} fmaps shape', final_fmaps[args.layer_name].shape)

	return final_fmaps

def train_model_THINGS2(args):
    """The function trains the encoding model using LinearRegression. X train 
    is THINGS2 dnn feature maps and Y train is the real THINGS EEG2 training 
    data.
    
    Parameters
    ----------
    args : Namespace
        Input arguments.

    Returns
    ----------
    reg: The trained LogisticRegression model.
    """

    import os
    import pickle
    import numpy as np
    from tqdm import tqdm
    from sklearn.linear_model import LinearRegression

    ### Load the training DNN feature maps ###
    # Load the training DNN feature maps directory
    dnn_train_dir = os.path.join('dataset', 'THINGS_EEG2', 'dnn_feature_maps')
    # Load the training DNN feature maps (16540,  num_feat)
    dnn_fmaps_train = np.load(os.path.join(dnn_train_dir, f'new_feature_maps_{args.num_feat}.npy'), 
                            allow_pickle=True)

    ### Load the training EEG data ###
    # Load the THINGS2 training EEG data directory
    eeg_train_dir = os.path.join('dataset', 'THINGS_EEG2', 'preprocessed_data_sleemory')
    # Iterate over THINGS2 subjects
    eeg_data_train = []
    for train_subj in tqdm(range(1,11), desc='load THINGS EEG2 subjects'):
        # Load the THINGS2 training EEG data
        with open(os.path.join(eeg_train_dir, 'sub-'+format(train_subj,'02'), 
                            'preprocessed_eeg_training.npy'), 'rb') as f:
            data = pickle.load(f)
            # Average the training EEG data across repetitions: (16540,num_ch,100)
            data = np.mean(data['preprocessed_eeg_data'], 1)
            # Drop the stim channel: (16540, num_ch-1, 100)
            data = np.delete(data, -1, axis=1)
            # Reshape the data: (16540, num_ch-1 x 100)
            data = np.reshape(data, (data.shape[0],-1))
            # Append individual data
            eeg_data_train.append(data)
            del data
    # Average the training EEG data across subjects: (16540, num_ch-1 x 100)
    eeg_data_train = np.mean(eeg_data_train, 0)
    print('eeg_data_train shape', eeg_data_train.shape)

    ### Train the encoding model ###
    # Train the encoding models
    reg = LinearRegression().fit(dnn_fmaps_train, eeg_data_train)
    return reg

def mvnn_mean(epoched_data):
    """Compute the covariance matrices of the EEG data (calculated for each
    time-point or epoch of each image condition), and then average them 
    across image conditions. The inverse of the resulting averaged covariance
    matrix is used to whiten the EEG data.

    Parameters
    ----------
    epoched_data : array of shape (image,channel,time)
        Epoched EEG data.

    Returns
    -------
    whitened_data : array of shape (image,channel,time)
        Whitened EEG data.
    """

    import numpy as np
    from sklearn.discriminant_analysis import _cov
    import scipy

    whitened_data = []
    # Notations
    img_cond = epoched_data.shape[0]
    num_ch = epoched_data.shape[1]
    num_time = epoched_data.shape[2]

    ### Compute the covariance matrices ###
    # Covariance matrix of shape:
    # EEG channels × EEG channels
    
    sigma = np.mean([_cov(epoched_data[:,:,t], shrinkage='auto') 
                     for t in range(num_time)],
                     axis=0)
    
    # Compute the inverse of the covariance matrix
    sigma_inv = scipy.linalg.fractional_matrix_power(sigma, -0.5) 
    # The matrix power is -0.5, which represents inverse square root
    
    ### Whiten the data ###
    whitened_data = (epoched_data.swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2)
    
    ### Output ###
    return whitened_data

def mvnn(all_epoched_data):
    """Compute the covariance matrices of the EEG data (calculated for each
    time-point or epoch of each image condition), and then average them 
    across image conditions. The inverse of the resulting averaged covariance
    matrix is used to whiten the EEG data.

    Parameters
    ----------
    epoched_data : list of arrays of shape (image,channel,time)
        Epoched EEG data.

    Returns
    -------
    whitened_data : list of arrays of shape (image,channel,time)
        Whitened EEG data.
    """

    import numpy as np
    from tqdm import tqdm
    from sklearn.discriminant_analysis import _cov
    import scipy

    ### Compute the covariance matrices ###
    tot_sigma = []
    for data in tqdm(all_epoched_data, desc='Whiten data across img conditions'): # Iterate over imgs
        
        # Notations
        num_rep = data.shape[0]
        num_ch = data.shape[1]
        num_time = data.shape[2]
        
        # Covariance matrix of shape (EEG channels × EEG channels)
        sigma = np.mean([_cov(data[:,:,t], shrinkage='auto') for t in range(num_time)],
                        axis=0)
        tot_sigma.append(sigma)
        
    # Average the covariance matrices across image conditions
    tot_sigma = np.array(tot_sigma)
    mean_sigma = np.mean(tot_sigma, axis=0)

    # # Compute the inverse of the covariance matrix
    sigma_inv = scipy.linalg.fractional_matrix_power(mean_sigma, -0.5)

    ### Whiten the data ###
    whitened_data = []
    for data in all_epoched_data: # Iterate over imgs     
        whitened_data.append((data.swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2))

    ### Output ###  
    return whitened_data